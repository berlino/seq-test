import torch
import triton
from torch.cuda.amp import custom_bwd, custom_fwd

import triton.language as tl
from einops import rearrange
from mingpt.kernels.utils import contiguous

def naive_lam_linear_attention(
    x, g,
    init_hidden,
    num_head,
    eps=1e-6,
    timenorm=False,
    actfn=None,
):
    if x.dtype == torch.bfloat16:
        print("Convert from bfloat16 to float32 in naive_lam_linear_attention")
        x, g, init_hidden = (
            x.to(torch.float32),
            g.to(torch.float32),
            init_hidden.to(torch.float32),
        )
    x = rearrange(x, "b l (n d) -> b n l d", n=num_head)
    g = rearrange(g, "b l (n d) -> b n l d", n=num_head)

    b, n, l, d = x.shape

    h = init_hidden  # b x n x d
    hstat_2nd = torch.zeros(b, n, 1, dtype=torch.float32, device=x.device) # prefix sum of square of deviations or square h

    h_l = []

    for i in range(l):
        if timenorm:
            num_elements = (i + 1) * d
            hstat_2nd = hstat_2nd + (h * h).sum(dim=-1, keepdim=True)
            rms_h = torch.sqrt(hstat_2nd / num_elements)
        else:
            hstat_2nd = (h * h).sum(dim=-1, keepdim=True)
            rms_h = torch.sqrt(hstat_2nd / d)
                
        x_i = x[:, :, i]  # b x n x d
        g_i = g[:, :, i]  # b x n x d

        rstd_h = 1.0 / torch.where(rms_h < eps, rms_h + eps, rms_h)
        # rstd_h = 1.0 / (rms_h + eps)
        h_hat = h * rstd_h  # b x n x d

        if actfn is None:
            new_h = (g_i * h_hat + (1-g_i) * x_i)  # b x n x d
        elif actfn == "relu":
            new_h = g_i * torch.relu(h_hat) + (1-g_i) * x_i
        elif actfn == "tanh":
            new_h = g_i * torch.tanh(h_hat) + (1-g_i) * x_i
        else:
            raise ValueError("actfn must be None, relu or tanh")
        h = new_h
        h_l.append(h)

    H = torch.stack(h_l, dim=2)  # b x n x l x d
    final_h = h_l[-1]
    return H, final_h

@triton.jit
def tl_tanh(x, eps=1e-6):
    denorm = tl.exp(x) + tl.exp(-x)
    denorm = tl.where(denorm < eps, denorm + eps, denorm)
    return (tl.exp(x) - tl.exp(-x)) / denorm

@triton.jit
def tl_dtanh(x):
    return 1.0 - tl_tanh(x) * tl_tanh(x)

@triton.jit
def tl_relu(x):
    return tl.where(x > 0, x, 0) 

@triton.jit
def tl_drelu(x):
    return tl.where(x > 0, 1.0, 0.0)

@triton.jit
def lam_linear_attention_fwd(
    X, G, HS, H, RMS_H, 
    BZ: tl.constexpr,
    NH: tl.constexpr,
    L: tl.constexpr,
    D: tl.constexpr,
    EPS: tl.constexpr,
    USE_TIMENORM: tl.constexpr,
    ACTFN: tl.constexpr, 
):
    b_idx = tl.program_id(0) // NH
    n_idx = tl.program_id(0) % NH

    X_ptr = X + b_idx * NH * L * D + n_idx * L * D + tl.arange(0, D)
    G_ptr = G + b_idx * NH * L * D + n_idx * L * D + tl.arange(0, D)
    H_ptr = H + b_idx * NH * L * D + n_idx * L * D + tl.arange(0, D)
    HS_ptr = HS + b_idx * NH * D + n_idx * D + tl.arange(0, D)
    RMS_H_ptr = RMS_H + b_idx * NH * L + n_idx * L + tl.arange(0, 1)

    h = tl.load(HS_ptr).to(tl.float32)
    hstat_2nd = tl.zeros([1], dtype=tl.float32)
    for t_idx in range(L):
        if USE_TIMENORM:
            hstat_2nd = hstat_2nd + tl.sum(h * h, axis=0)
            rms_h = tl.sqrt(hstat_2nd / (t_idx + 1.) / D)
        else:
            hstat_2nd = hstat_2nd * 0.0 + tl.sum(h * h, axis=0)
            rms_h = tl.sqrt(hstat_2nd / D)

        x = tl.load(X_ptr).to(tl.float32)
        g = tl.load(G_ptr).to(tl.float32)

        rstd_h = 1.0 / tl.where(rms_h < EPS, rms_h + EPS, rms_h)
        # rstd_h = 1.0 / (rms_h + EPS)
        tl.store(RMS_H_ptr, rstd_h.to(RMS_H_ptr.dtype.element_ty))
        h_hat = h * rstd_h

        if ACTFN == "tanh":
            h = g * tl_tanh(h_hat) + x * (1.0 - g)
        elif ACTFN == "relu":
            h = g * tl_relu(h_hat) + x * (1.0 - g)
        else:
            h = h_hat * g + x * (1.0 - g) 
        tl.store(H_ptr, h.to(H_ptr.dtype.element_ty))

        X_ptr += D
        G_ptr += D
        H_ptr += D
        RMS_H_ptr += 1


@triton.jit
def lam_linear_attention_bwd(
        X, G, HS, 
        H, RSTD_H, dH,
        dX, dG,
        BZ: tl.constexpr, 
        NH: tl.constexpr, 
        L: tl.constexpr, 
        D: tl.constexpr,
        EPS: tl.constexpr,
        USE_TIMENORM: tl.constexpr,
        ACTFN: tl.constexpr,
):
    b_idx = tl.program_id(0) // NH
    n_idx = tl.program_id(0) % NH

    X_ptr = X + b_idx * NH * L * D + n_idx * L * D + (L - 1) * D + tl.arange(0, D)
    G_ptr = G + b_idx * NH * L * D + n_idx * L * D + (L - 1) * D + tl.arange(0, D)
    H_ptr = H + b_idx * NH * L * D + n_idx * L * D + (L - 2) * D + tl.arange(0, D)
    HS_ptr = HS + b_idx * NH * D + n_idx * D + tl.arange(0, D)
    RSTD_H_ptr = RSTD_H + b_idx * NH * L + n_idx * L + (L - 1) + tl.arange(0, 1)
    dX_ptr = dX + b_idx * NH * L * D + n_idx * L * D + (L - 1) * D + tl.arange(0, D)
    dG_ptr = dG + b_idx * NH * L * D + n_idx * L * D + (L - 1) * D + tl.arange(0, D)
    dH_ptr = dH + b_idx * NH * L * D + n_idx * L * D + (L - 1) * D + tl.arange(0, D)

    # dh1 from do, dh2 from next token
    dh2 = tl.zeros([D], dtype=tl.float32)

    if USE_TIMENORM:
        # from next token, very similar to dh2
        dh_2nd_history_acc = tl.zeros([D], dtype=tl.float32)

    for t_idx in range(L-1, -1, -1):
        x = tl.load(X_ptr)
        g = tl.load(G_ptr)

        if t_idx == 0:
            prev_h = tl.load(HS_ptr).to(tl.float32)
        else:
            prev_h = tl.load(H_ptr).to(tl.float32)

        dh1 = tl.load(dH_ptr)

        rstd_h = tl.load(RSTD_H_ptr)
        h_hat = prev_h * rstd_h

        if ACTFN == "tanh":
            h_tilde = tl_tanh(h_hat)
        elif ACTFN == "relu":
            h_tilde = tl_relu(h_hat)
        else:
            h_tilde = h_hat
        
        dh = dh1 + dh2
        dx = (1.0 - g) * dh
        dg = h_tilde * dh - dh * x
        tl.store(dX_ptr, dx.to(dX_ptr.dtype.element_ty))
        tl.store(dG_ptr, dg.to(dG_ptr.dtype.element_ty))

        dh_tilde = dh * g 
        if ACTFN == "tanh":
            dh_hat = dh_tilde * tl_dtanh(h_hat)
        elif ACTFN == "relu":
            dh_hat = dh_tilde * tl_drelu(h_hat)
        else:
            dh_hat = dh_tilde

        if USE_TIMENORM:
            dpartial_h_1 = dh_hat * rstd_h
            
            rstd_h_square = rstd_h * rstd_h
            drms_prev_h = - tl.sum(dh_hat * prev_h, axis=0) * rstd_h_square
            dmean_square = 0.5 * drms_prev_h * rstd_h
            dh_2nd_history = dmean_square / (t_idx + 1.) / D
            d_partial_h_2 = 2. * prev_h * dh_2nd_history + 2. * prev_h * dh_2nd_history_acc
            dh_2nd_history_acc += dh_2nd_history

            dh2_prev = dpartial_h_1 + d_partial_h_2
            dh2 = dh2_prev
        else:
            # from numerator of h_t / rms(h_{t-1})
            dpartial_h_1 = dh_hat * rstd_h

            # from denominator of h_t / rms(h_{t-1})
            # 1 / x, note that drms_prev_h is a scalar
            rstd_h_square = rstd_h * rstd_h
            drms_prev_h = - (tl.sum(dh_hat * prev_h, axis=0)) * rstd_h_square 
            # sqrt(x)
            dmean_square = 0.5 * drms_prev_h * rstd_h 
            # mean_square
            dpartial_h_2 = 2. * prev_h * dmean_square / D

            dh2_prev = dpartial_h_1 + dpartial_h_2
            dh2 = dh2_prev

        X_ptr -= D
        G_ptr -= D
        H_ptr -= D
        RSTD_H_ptr -= 1
        dX_ptr -= D
        dG_ptr -= D
        dH_ptr -= D


class Lam1DLinearAttention(torch.autograd.Function):
    @staticmethod
    @contiguous
    @custom_fwd
    def forward(ctx, X, G, HS):
        bz, nh, l, d = X.shape
        eps = 1e-6

        timenorm = True
        actfn = "tanh"

        grid = (bz * nh, 1)

        H = torch.empty(bz, nh, l, d, dtype=torch.float32, device=X.device)
        RSTD_H = torch.empty(bz, nh, l, dtype=torch.float32, device=X.device)
        lam_linear_attention_fwd[grid](
            X, G, HS, H, RSTD_H,
            bz, nh, l, d,
            eps, timenorm, actfn)

        ctx.save_for_backward(X, G, HS, H, RSTD_H)
        ctx.d = d
        ctx.eps = eps
        ctx.actfn = actfn
        ctx.timenorm = timenorm
        return H.to(X.dtype)

    @staticmethod
    @contiguous
    @custom_bwd
    def backward(ctx, dH):
        bz, nh, l, d = dH.shape

        X, G, HS, H, RSTD_H, = ctx.saved_tensors
        eps = ctx.eps
        actfn = ctx.actfn
        timenorm = ctx.timenorm

        dX = torch.empty_like(X, dtype=torch.float32)
        dG = torch.empty_like(G, dtype=torch.float32)

        grid = (bz * nh, 1)
        lam_linear_attention_bwd[grid](
            X, G, HS, H, RSTD_H, dH,
            dX, dG, 
            bz, nh, l, d,
            eps, timenorm, actfn)
        return dX.to(X.dtype), dG.to(G.dtype), None

apply_1d_lam_linear_attention = Lam1DLinearAttention.apply

def lam_1d_linear_attention(x, g, num_head, hs=None):
    bz, d = x.shape[0], x.shape[-1]
    if hs is None:
        hs = x.new_zeros(bz, num_head, d, dtype=torch.float32)

    x = rearrange(x, 'b l (n d) -> b n l d', n=num_head)
    g = rearrange(g, 'b l (n d) -> b n l d', n=num_head)
    h = apply_1d_lam_linear_attention(x, g, hs)
    he = h[:, :, -1] # b x n x d
    return h, he


if __name__ == "__main__":
    # dtype = torch.bfloat16
    dtype = torch.float32
    device = torch.device("cuda:0")
    bs, l, nh, d = 2, 1024, 4, 32
    x = torch.randn(bs, l, nh * d, dtype=dtype, device=device, requires_grad=True)
    g = torch.randn(bs, l, nh * d, dtype=dtype, device=device, requires_grad=True)
    hs = torch.randn(bs, nh, d, dtype=dtype, device=device, requires_grad=True)
    eps = 1e-6
    timenorm = True
    actfn = "tanh"
    dH = torch.randn(bs, nh, l, d, dtype=dtype, device=device)

    H1, HE1 = naive_lam_linear_attention(x, g.sigmoid(), hs, nh, eps=eps, timenorm=timenorm, actfn=actfn)
    H1.backward(dH)

    x_grad_1, g_grad_1, hs_grad_1 = x.grad, g.grad, hs.grad
    x.grad, g.grad, hs.grad = None, None, None

    H2, HE2 = lam_1d_linear_attention(x, g.sigmoid(), nh, hs)
    H2.backward(dH)

    x_grad_2, g_grad_2, hs_grad_2 = x.grad, g.grad, hs.grad

    print("forward H diff", (H1 - H2).abs().max())
    print("forward HE diff", (HE1 - HE2).abs().max())
    print("backward x diff", (x_grad_1 - x_grad_2).abs().max())
    print("backward g diff", (g_grad_1 - g_grad_2).abs().max())
