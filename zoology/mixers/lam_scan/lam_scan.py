import torch
import triton
from torch.cuda.amp import custom_bwd, custom_fwd

import triton.language as tl
from einops import rearrange
from mingpt.kernels.utils import contiguous


def naive_lam_linear_attention(
    q, k, v, gk,
    init_hidden,
    num_head,
    eps=1e-6,
    is_rms=True,
    timenorm=False,
    residual=False,
    cummax=False,
    return_mv=False,
):
    assert not (residual and cummax), "residual and cummax cannot be paired"
    if q.dtype == torch.bfloat16:
        print("Convert from bfloat16 to float32 in naive_lam_linear_attention")
        q, k, v, gk, init_hidden = (
            q.to(torch.float32),
            k.to(torch.float32),
            v.to(torch.float32),
            gk.to(torch.float32),
            init_hidden.to(torch.float32),
        )
    q = rearrange(q, "b l (n d) -> b n l d", n=num_head)
    k = rearrange(k, "b l (n d) -> b n l d", n=num_head)
    v = rearrange(v, "b l (n d) -> b n l d", n=num_head)
    gk = rearrange(gk, "b l (n d) -> b n l d", n=num_head)

    b, n, l, d_k = q.shape
    d_v = v.shape[-1]

    h = init_hidden  # b x n x d_v x d_k
    h_prime = init_hidden  # b x n x d_v x d_k
    hstat_1st=  torch.zeros(b, n, d_v, 1, dtype=torch.float32, device=q.device) # standard prefix sum of h
    hstat_2nd = torch.zeros(b, n, d_v, 1, dtype=torch.float32, device=q.device) # prefix sum of square of deviations or square h

    mean_l = []
    rstd_l = []
    o_l = []

    for i in range(l):
        if timenorm:
            num_elements = (i + 1) * d_k
            if is_rms:
                h_tilde = h
                hstat_2nd = hstat_2nd + (h * h).sum(dim=-1, keepdim=True)
            else:
                if i == 0:
                    prev_mean_h = torch.zeros_like(hstat_1st)
                else:
                    prev_mean_h = hstat_1st / (i * d_k)
                hstat_1st = hstat_1st + h.sum(dim=-1, keepdim=True)
                mean_h = hstat_1st / num_elements
                mean_l.append(mean_h)
                h_tilde = h - mean_h
                hstat_2nd = hstat_2nd + ((h - prev_mean_h) * h_tilde).sum(dim=-1, keepdim=True)
            rms_h = torch.sqrt(hstat_2nd / num_elements)
        else:
            if is_rms:
                h_tilde = h
                hstat_2nd = (h * h).sum(dim=-1, keepdim=True)
            else:
                hstat_1st = h.sum(dim=-1, keepdim=True)
                mean_h = hstat_1st / d_k
                mean_l.append(mean_h)
                h_tilde = h - mean_h
                hstat_2nd = ((h - mean_h) * (h - mean_h)).sum(dim=-1, keepdim=True)
            rms_h = torch.sqrt(hstat_2nd / d_k)
                
        q_i = q[:, :, i]  # b x n x d_k
        k_i = k[:, :, i]  # b x n x d_k
        v_i = v[:, :, i]  # b x n x d_v
        gk_i = gk[:, :, i]  # b x n x d_k

        # rstd_h = 1.0 / (rms_h + eps)
        rstd_h = 1.0 / torch.where(rms_h < eps, rms_h + eps, rms_h)
        rstd_l.append(rstd_h)
        h_hat = h_tilde * rstd_h  # b x n x d_v x d_k

        new_h = (h_hat * gk_i[:, :, None, :] + k_i[:, :, None, :] * v_i[:, :, :, None])  # b x n x d_v x d_k

        if residual:
            new_h += h

        if cummax:
            h_prime_mask = h_prime > new_h
            h_prime = h_prime_mask * h_prime + ~h_prime_mask * new_h 
            o_i = (q_i[:, :, None, :] * h_prime).sum(-1)  # b x n x d_v
        else:
            o_i = (q_i[:, :, None, :] * new_h).sum(-1)  # b x n x d_v
        o_l.append(o_i)
        h = new_h

    final_h = h
    o = torch.stack(o_l, dim=1)  # b x l x n x d_v
    o = rearrange(o, "b l n d -> b l (n d)")

    if return_mv and is_rms:
        rstd = torch.cat(rstd_l, dim=-1)
        rstd = rearrange(rstd, "b n d l -> b l (n d)")
        return o, final_h, None, rstd
    elif return_mv and not is_rms:
        mean = torch.cat(mean_l, dim=-1)
        rstd = torch.cat(rstd_l, dim=-1)
        mean = rearrange(mean, "b n d l -> b l (n d)")
        rstd = rearrange(rstd, "b n d l -> b l (n d)")
        return o, final_h, mean, rstd
    else:
        return o, final_h


@triton.jit
def lam_linear_attention_fwd(
    Q, K, V, GK, HS, 
    O, H, HE, MEAN, RSTD, CUMMAX_MASK, H_PRIME,
    BZ: tl.constexpr,
    NH: tl.constexpr,
    L: tl.constexpr,
    D_K: tl.constexpr,
    D_V: tl.constexpr,
    B_V: tl.constexpr,
    EPS: tl.constexpr,
    IS_RMS: tl.constexpr,
    USE_TIMENORM: tl.constexpr,
    RESIDUAL: tl.constexpr,
    CUMMAX: tl.constexpr
):
    b_idx = tl.program_id(0) // NH
    n_idx = tl.program_id(0) % NH
    v_block_idx = tl.program_id(1) 

    Q_ptr = Q + b_idx * NH * L * D_K + n_idx * L * D_K + tl.arange(0, D_K)
    K_ptr = K + b_idx * NH * L * D_K + n_idx * L * D_K + tl.arange(0, D_K)
    GK_ptr = GK + b_idx * NH * L * D_K + n_idx * L * D_K + tl.arange(0, D_K)
    V_ptr = V + b_idx * NH * L * D_V + n_idx * L * D_V + v_block_idx * B_V + tl.arange(0, B_V)

    HS_ptr = HS + b_idx * NH * D_V * D_K + n_idx * D_V * D_K + v_block_idx * B_V * D_K + tl.arange(0, B_V)[:, None] * D_K + tl.arange(0, D_K)[None, :]
    HE_ptr = HE + b_idx * NH * D_V * D_K + n_idx * D_V * D_K + v_block_idx * B_V * D_K + tl.arange(0, B_V)[:, None] * D_K + tl.arange(0, D_K)[None, :]
    H_ptr = H + b_idx * NH * L * D_V * D_K + n_idx * L * D_V * D_K + v_block_idx * B_V * D_K + tl.arange(0, B_V)[:, None] * D_K + tl.arange(0, D_K)[None, :]

    O_ptr = O + b_idx * NH * L * D_V + n_idx * L * D_V + v_block_idx * B_V + tl.arange(0, B_V)
    RSTD_ptr = RSTD + b_idx * NH * L * D_V + n_idx * L * D_V + v_block_idx * B_V + tl.arange(0, B_V)
    if not IS_RMS:
        M_ptr = MEAN + b_idx * NH * L * D_V + n_idx * L * D_V + v_block_idx * B_V + tl.arange(0, B_V)
    
    if CUMMAX:
        CUMMAX_MASK_ptr = CUMMAX_MASK + b_idx * NH * L * D_V * D_K + n_idx * L * D_V * D_K + v_block_idx * B_V * D_K + tl.arange(0, B_V)[:, None] * D_K + tl.arange(0, D_K)[None, :]
        H_PRIME_ptr = H_PRIME + b_idx * NH * L * D_V * D_K + n_idx * L * D_V * D_K + v_block_idx * B_V * D_K + tl.arange(0, B_V)[:, None] * D_K + tl.arange(0, D_K)[None, :]

    h = tl.load(HS_ptr).to(tl.float32)
    if CUMMAX:
        h_prime = tl.load(HS_ptr).to(tl.float32)
    hstat_1st = tl.zeros([B_V], dtype=tl.float32) 
    hstat_2nd = tl.zeros([B_V], dtype=tl.float32)
    for t_idx in range(L):
        if USE_TIMENORM:
            if IS_RMS:
                h_tilde = h
                hstat_2nd = hstat_2nd + tl.sum(h * h, axis=1)
            else:
                prev_mean_h = hstat_1st / (t_idx * D_K + EPS)  # handle the case of t_idx = 0
                # ratio = (t_idx * D_K + EPS) / ((t_idx + 1.) * D_K)
                # mean_h = prev_mean_h * ratio + tl.sum(h, axis=1) / ((t_idx + 1.) * D_K)
                hstat_1st = hstat_1st + tl.sum(h, axis=1)
                mean_h = hstat_1st / ((t_idx + 1.) * D_K)
                tl.store(M_ptr, mean_h.to(M_ptr.dtype.element_ty))
                h_tilde = h - mean_h[:, None]
                h_bar = h - prev_mean_h[:, None]
                hstat_2nd = hstat_2nd + tl.sum(h_bar * h_tilde, axis=1)

            rms_h = tl.sqrt(hstat_2nd / (t_idx + 1.) / D_K)

        else:
            if IS_RMS:
                h_tilde = h
                hstat_2nd = hstat_2nd * 0.0 + tl.sum(h * h, axis=1)
            else:
                hstat_1st = hstat_1st * 0.0 + tl.sum(h, axis=1)
                mean_h = tl.sum(hstat_1st, axis=1) / D_K
                tl.store(M_ptr, mean_h.to(M_ptr.dtype.element_ty))
                h_tilde = h - mean_h[:, None]
                hstat_2nd = hstat_2nd * 0.0 + tl.sum(h_tilde * h_tilde, axis=1)

            rms_h = tl.sqrt(hstat_2nd / D_K)

        q = tl.load(Q_ptr)
        k = tl.load(K_ptr).to(tl.float32)
        v = tl.load(V_ptr).to(tl.float32)
        gk = tl.load(GK_ptr).to(tl.float32)

        rstd_h = 1.0 / tl.where(rms_h < EPS, rms_h + EPS, rms_h)
        tl.store(RSTD_ptr, rstd_h.to(RSTD_ptr.dtype.element_ty))
        h_hat = h_tilde * rstd_h[:, None] 

        if RESIDUAL:
            h = h_hat * gk[None, :] + k[None, :] * v[:, None] + h
        else:
            h = h_hat * gk[None, :] + k[None, :] * v[:, None]
        tl.store(H_ptr, h.to(H_ptr.dtype.element_ty))
        
        if CUMMAX:
            h_prime_mask = h_prime > h
            h_prime = h_prime_mask * h_prime + ~h_prime_mask * h
            tl.store(CUMMAX_MASK_ptr, h_prime_mask)
            tl.store(H_PRIME_ptr, h_prime.to(H_PRIME_ptr.dtype.element_ty))
            o = tl.sum(q[None, :] * h_prime, axis=1)
        else:
            o = tl.sum(h * q[None, :], axis=1)
        tl.store(O_ptr, o)

        Q_ptr += D_K
        K_ptr += D_K
        V_ptr += D_V
        GK_ptr += D_K
        O_ptr += D_V
        H_ptr += D_V * D_K
        RSTD_ptr += D_V
        if not IS_RMS:
            M_ptr += D_V
        if CUMMAX:
            CUMMAX_MASK_ptr += D_V * D_K
            H_PRIME_ptr += D_V * D_K

    tl.store(HE_ptr, h.to(HE_ptr.dtype.element_ty))


@triton.jit
def lam_linear_attention_bwd(
        Q, K, V, GK, HS, H, MEAN, RSTD, CUMMAX_MASK, H_PRIME,
        dO, dMEAN, dRSTD,
        dQ, dK, dV, dGK, dHS,
        BZ: tl.constexpr, 
        NH: tl.constexpr, 
        L: tl.constexpr, 
        D_K: tl.constexpr, 
        D_V: tl.constexpr,
        B_V: tl.constexpr, 
        EPS: tl.constexpr,
        IS_RMS: tl.constexpr,
        USE_TIMENORM: tl.constexpr,
        RESIDUAL: tl.constexpr,
        CUMMAX: tl.constexpr
):
    b_idx = tl.program_id(0) // NH
    n_idx = tl.program_id(0) % NH
    v_block_idx = tl.program_id(1)

    Q_ptr = Q + b_idx * NH * L * D_K + n_idx * L * D_K + (L - 1) * D_K + tl.arange(0, D_K)
    K_ptr = K + b_idx * NH * L * D_K + n_idx * L * D_K + (L - 1) * D_K + tl.arange(0, D_K)
    GK_ptr = GK + b_idx * NH * L * D_K + n_idx * L * D_K + (L - 1) * D_K + tl.arange(0, D_K)

    dQ_ptr = dQ + v_block_idx * BZ * NH * L * D_K + b_idx * NH * L * D_K + n_idx * L * D_K + (L - 1) * D_K + tl.arange(0, D_K)
    dK_ptr = dK + v_block_idx * BZ * NH * L * D_K + b_idx * NH * L * D_K + n_idx * L * D_K + (L - 1) * D_K + tl.arange(0, D_K)
    dGK_ptr = dGK + v_block_idx * BZ * NH * L * D_K + b_idx * NH * L * D_K + n_idx * L * D_K + (L - 1) * D_K + tl.arange(0, D_K)

    V_ptr = V + b_idx * NH * L * D_V + n_idx * L * D_V + (L - 1) * D_V + v_block_idx * B_V + tl.arange(0, B_V)
    dV_ptr = dV + b_idx * NH * L * D_V + n_idx * L * D_V + (L - 1) * D_V + v_block_idx * B_V + tl.arange(0, B_V)

    H_ptr = H + b_idx * NH * L * D_V * D_K + n_idx * L * D_V * D_K + (L - 2) * D_V * D_K + v_block_idx * B_V * D_K + tl.arange(0, B_V)[:, None] * D_K + tl.arange(0, D_K)[None, :]

    HS_ptr = HS + b_idx * NH * D_V * D_K + n_idx * D_V * D_K + v_block_idx * B_V * D_K + tl.arange(0, B_V)[:, None] * D_K + tl.arange(0, D_K)[None, :]
    dHS_ptr = dHS + b_idx * NH * D_V * D_K + n_idx * D_V * D_K + v_block_idx * B_V * D_K + tl.arange(0, B_V)[:, None] * D_K + tl.arange(0, D_K)[None, :]

    dO_ptr = dO + b_idx * NH * L * D_V + n_idx * L * D_V + (L - 1) * D_V + v_block_idx * B_V + tl.arange(0, B_V)
    RSTD_ptr = RSTD + b_idx * NH * L * D_V + n_idx * L * D_V + (L - 1) * D_V + v_block_idx * B_V + tl.arange(0, B_V)
    dRSTD_ptr = dRSTD + b_idx * NH * L * D_V + n_idx * L * D_V + (L - 1) * D_V + v_block_idx * B_V + tl.arange(0, B_V)
    if not IS_RMS:
        M_ptr = MEAN + b_idx * NH * L * D_V + n_idx * L * D_V + (L - 1) * D_V + v_block_idx * B_V + tl.arange(0, B_V)
        dM_ptr = dMEAN + b_idx * NH * L * D_V + n_idx * L * D_V + (L - 1) * D_V + v_block_idx * B_V + tl.arange(0, B_V)
    
    if CUMMAX:
        CUMMAX_MASK_ptr = CUMMAX_MASK + b_idx * NH * L * D_V * D_K + n_idx * L * D_V * D_K + (L - 1) * D_V * D_K + v_block_idx * B_V * D_K + tl.arange(0, B_V)[:, None] * D_K + tl.arange(0, D_K)[None, :]
        H_PRIME_ptr = H_PRIME + b_idx * NH * L * D_V * D_K + n_idx * L * D_V * D_K + (L - 1) * D_V * D_K + v_block_idx * B_V * D_K + tl.arange(0, B_V)[:, None] * D_K + tl.arange(0, D_K)[None, :]

    # dh1 from do, dh2 from next token
    dh2 = tl.zeros([B_V, D_K], dtype=tl.float32)

    if USE_TIMENORM:
        # from next token, very similar to dh2
        dh_2nd_history_acc = tl.zeros([B_V, D_K], dtype=tl.float32)
        if not IS_RMS:

            dh_1st_history_acc = tl.zeros([B_V, D_K], dtype=tl.float32)
    if CUMMAX:
        # from next token
        dh_prime_acc = tl.zeros([B_V, D_K], dtype=tl.float32)

    for t_idx in range(L-1, 0, -1):
        q = tl.load(Q_ptr)
        k = tl.load(K_ptr).to(tl.float32)
        v = tl.load(V_ptr).to(tl.float32)
        gk = tl.load(GK_ptr).to(tl.float32)
        do = tl.load(dO_ptr)

        prev_h = tl.load(H_ptr)
        rstd_h = tl.load(RSTD_ptr)
        drstd_h = tl.load(dRSTD_ptr)

        if not IS_RMS:
            mean_h = tl.load(M_ptr)
            h_tilde = prev_h - mean_h[:, None]
        else:
            h_tilde = prev_h
        h_hat = h_tilde  * rstd_h[:, None]

        if RESIDUAL:
            h = h_hat * gk[None, :] + k[None, :] * v[:, None] + prev_h
        else:
            h = h_hat * gk[None, :] + k[None, :] * v[:, None]
        
        if CUMMAX:
            h_prime_mask = tl.load(CUMMAX_MASK_ptr).to(tl.int1)
            dh_prime = q[None, :] * do[:, None] + dh_prime_acc # for current step
            dh1 = dh_prime * ~h_prime_mask
            dh_prime_acc = dh_prime * h_prime_mask
            h_prime = tl.load(H_PRIME_ptr)
        else:
            dh1 = q[None, :] * do[:, None]
            h_prime = h
        dh = dh1 + dh2

        dv = tl.sum(dh * k[None, :], axis=1)
        dk = tl.sum(dh * v[:, None], axis=0)
        dq = tl.sum(h_prime * do[:, None], axis=0)
        dgk = tl.sum(dh * h_hat, axis=0)

        tl.store(dQ_ptr, dq.to(dQ_ptr.dtype.element_ty))
        tl.store(dK_ptr, dk.to(dK_ptr.dtype.element_ty))
        tl.store(dV_ptr, dv.to(dV_ptr.dtype.element_ty))
        tl.store(dGK_ptr, dgk.to(dGK_ptr.dtype.element_ty))

        dh_hat = dh * gk[None, :]

        if USE_TIMENORM:
            if not IS_RMS:
                dmean_h = tl.load(dM_ptr)
                dmean_h_1st_history = - (tl.sum(dh_hat * rstd_h[:, None], axis=1) + dmean_h) 
                dh_1st_history = (dmean_h_1st_history / (t_idx + 1.) / D_K)[:, None]
                dpartial_h_1 = dh_hat * rstd_h[:, None] + dh_1st_history + dh_1st_history_acc
                dh_1st_history_acc += dh_1st_history
            else:
                dpartial_h_1 = dh_hat * rstd_h[:, None]
            
            rstd_h_square = rstd_h * rstd_h
            drms_prev_h = - (tl.sum(dh_hat * h_tilde, axis=1) + drstd_h) * rstd_h_square
            dmean_square = 0.5 * drms_prev_h * rstd_h
            dh_2nd_history = dmean_square[:, None] / (t_idx + 1.) / D_K
            d_partial_h_2 = 2. * prev_h * dh_2nd_history + 2. * prev_h * dh_2nd_history_acc
            dh_2nd_history_acc += dh_2nd_history

            dh2_prev = dpartial_h_1 + d_partial_h_2
            if RESIDUAL:
                dh2_prev += dh
            dh2 = dh2_prev
        else:
            # from numerator of h_t / rms(h_{t-1})
            if not IS_RMS:
                dmean_h = tl.load(dM_ptr)
                dpartial_h_1 = dh_hat * rstd_h[:, None] - (tl.sum(dh_hat * rstd_h[:, None], axis=1) / D_K)[:, None] + (dmean_h / D_K)[:, None]
            else:
                dpartial_h_1 = dh_hat * rstd_h[:, None]

            # from denominator of h_t / rms(h_{t-1})
            # 1 / x, note that drms_prev_h is a scalar
            rstd_h_square = rstd_h * rstd_h
            drms_prev_h = - (tl.sum(dh_hat * h_tilde, axis=1) + drstd_h) * rstd_h_square 
            # sqrt(x)
            dmean_square = 0.5 * drms_prev_h * rstd_h 
            # mean_square
            dpartial_h_2 = 2. * prev_h * dmean_square[:, None] / D_K

            dh2_prev = dpartial_h_1 + dpartial_h_2
            if RESIDUAL:
                dh2_prev += dh
            dh2 = dh2_prev

        Q_ptr -= D_K
        K_ptr -= D_K
        V_ptr -= D_V
        dO_ptr -= D_V
        GK_ptr -= D_K
        H_ptr -= D_V * D_K
        dQ_ptr -= D_K
        dK_ptr -= D_K
        dV_ptr -= D_V
        dGK_ptr -= D_K
        RSTD_ptr -= D_V
        dRSTD_ptr -= D_V

        if not IS_RMS:
            M_ptr -= D_V
            dM_ptr -= D_V
        
        if CUMMAX:
            CUMMAX_MASK_ptr -= D_V * D_K
            H_PRIME_ptr -= D_V * D_K

    # handle the first token, h comes for HE
    q = tl.load(Q_ptr)
    k = tl.load(K_ptr).to(tl.float32)
    v = tl.load(V_ptr).to(tl.float32)
    gk = tl.load(GK_ptr).to(tl.float32)

    do = tl.load(dO_ptr)
    prev_h = tl.load(HS_ptr)
    rstd_h = tl.load(RSTD_ptr)
    drstd_h = tl.load(dRSTD_ptr)

    if not IS_RMS:
        mean_h = tl.load(M_ptr)
        h_tilde = prev_h - mean_h[:, None]
    else:
        h_tilde = prev_h
    h_hat = h_tilde  * rstd_h[:, None]

    if RESIDUAL:
        h = h_hat * gk[None, :] + k[None, :] * v[:, None] + prev_h
    else:
        h = h_hat * gk[None, :] + k[None, :] * v[:, None]
    
    if CUMMAX:
        h_prime_mask = tl.load(CUMMAX_MASK_ptr).to(tl.int1)
        dh_prime = q[None, :] * do[:, None] + dh_prime_acc # for current step
        dh1 = dh_prime * ~h_prime_mask
        dh_prime_acc = dh_prime * h_prime_mask
        h_prime = tl.load(H_PRIME_ptr)
    else:
        dh1 = q[None, :] * do[:, None]
        h_prime = h
    dh = dh1 + dh2

    dv = tl.sum(dh * k[None, :], axis=1)
    dk = tl.sum(dh * v[:, None], axis=0)
    dq = tl.sum(h_prime * do[:, None], axis=0)
    dgk = tl.sum(dh * h_hat, axis=0)

    tl.store(dQ_ptr, dq.to(dQ_ptr.dtype.element_ty))
    tl.store(dK_ptr, dk.to(dK_ptr.dtype.element_ty))
    tl.store(dV_ptr, dv.to(dV_ptr.dtype.element_ty))
    tl.store(dGK_ptr, dgk.to(dGK_ptr.dtype.element_ty))

    dh_hat = dh * gk[None, :]

    if USE_TIMENORM:
        if not IS_RMS:
            dmean_h = tl.load(dM_ptr)
            dmean_h_1st_history = - (tl.sum(dh_hat * rstd_h[:, None], axis=1) + dmean_h) 
            dh_1st_history = (dmean_h_1st_history / (t_idx + 1.) / D_K)[:, None]
            dpartial_h_1 = dh_hat * rstd_h[:, None] + dh_1st_history + dh_1st_history_acc
            dh_1st_history_acc += dh_1st_history
        else:
            dpartial_h_1 = dh_hat * rstd_h[:, None]
        
        rstd_h_square = rstd_h * rstd_h
        drms_prev_h = - (tl.sum(dh_hat * h_tilde, axis=1) + drstd_h) * rstd_h_square
        dmean_square = 0.5 * drms_prev_h * rstd_h
        dh_2nd_history = dmean_square[:, None] / (t_idx + 1.) / D_K
        d_partial_h_2 = 2. * prev_h * dh_2nd_history + 2. * prev_h * dh_2nd_history_acc
        dh_2nd_history_acc += dh_2nd_history

        dh2_prev = dpartial_h_1 + d_partial_h_2
        if RESIDUAL:
            dh2_prev += dh
    else:
        if not IS_RMS:
            dmean_h = tl.load(dM_ptr)
            dpartial_h_1 = dh_hat * rstd_h[:, None] - (tl.sum(dh_hat * rstd_h[:, None], axis=1) / D_K)[:, None] + (dmean_h / D_K)[:, None]
        else:
            dpartial_h_1 = dh_hat * rstd_h[:, None]

        # 1 / x, note that drms_prev_h is a scalar
        rstd_h_square = rstd_h * rstd_h
        drms_prev_h = - (tl.sum(dh_hat * h_tilde, axis=1) + drstd_h) * rstd_h_square 
        # sqrt(x)
        dmean_square = 0.5 * drms_prev_h * rstd_h 
        # mean_square
        dpartial_h_2 = 2. * prev_h * dmean_square[:, None] / D_K
    
        dh2_prev = dpartial_h_1 + dpartial_h_2
        if RESIDUAL:
            dh2_prev += dh

    if CUMMAX:
        dHS = dh2_prev + dh_prime_acc
    else:
        dHS = dh2_prev
    tl.store(dHS_ptr, dHS.to(dHS_ptr.dtype.element_ty))


class LamLinearAttention(torch.autograd.Function):
    @staticmethod
    @contiguous
    @custom_fwd
    def forward(ctx, Q, K, V, GK, HS, HE):
        bz, nh, l, d_k = Q.shape
        d_v = V.shape[-1]
        block_size_v = 64 
        eps = 1e-6

        is_rms = True
        timenorm = True
        residual = False
        cummax = True

        num_block_v = d_v // block_size_v
        assert num_block_v * block_size_v == d_v
        grid = (bz * nh, num_block_v)

        O = torch.empty_like(V)
        H = torch.empty(bz, nh, l, d_v, d_k, dtype=torch.float32, device=Q.device)
        RSTD = torch.empty_like(V, dtype=torch.float32)

        if not is_rms:
            MEAN = torch.empty_like(V, dtype=torch.float32)
        else:
            MEAN = None

        if cummax:
            CUMMAX_MASK = torch.empty(bz, nh, l, d_v, d_k, dtype=torch.bool, device=Q.device)
            H_PRIME = torch.empty_like(H)
        else:
            CUMMAX_MASK = None
            H_PRIME = None

        lam_linear_attention_fwd[grid](
            Q, K, V, GK, HS, 
            O, H, HE, MEAN, RSTD, CUMMAX_MASK, H_PRIME,
            bz, nh, l, d_k, d_v, block_size_v, 
            eps, is_rms, timenorm, residual, cummax)

        ctx.save_for_backward(Q, K, V, GK, HS, H, MEAN, RSTD, CUMMAX_MASK, H_PRIME)
        ctx.d_k = d_k
        ctx.block_size_v = block_size_v
        ctx.eps = eps
        ctx.is_rms = is_rms
        ctx.timenorm = timenorm
        ctx.residual = residual
        ctx.cummax = cummax
        return O, MEAN, RSTD

    @staticmethod
    @contiguous
    @custom_bwd
    def backward(ctx, dO, dMEAN, dRSTD):
        bz, nh, l, d_v = dO.shape

        Q, K, V, GK, HS, H, MEAN, RSTD, CUMMAX_MASK, H_PRIME = ctx.saved_tensors
        d_k = ctx.d_k
        eps = ctx.eps
        is_rms = ctx.is_rms
        timenorm = ctx.timenorm
        residual = ctx.residual
        cummax = ctx.cummax

        block_size_v = min(ctx.block_size_v, d_v)
        num_block_v = d_v // block_size_v

        dQ = Q.new_empty(num_block_v, bz, nh, l, d_k)
        dK = K.new_empty(num_block_v, bz, nh, l, d_k)
        dGK = GK.new_empty(num_block_v, bz, nh, l, d_k)
        dV = V.new_empty(bz, nh, l, d_v)
        dHS = HS.new_empty(bz, nh, d_v, d_k)

        grid = (bz * nh, num_block_v)
        lam_linear_attention_bwd[grid](
            Q, K, V, GK, HS, H, MEAN, RSTD, CUMMAX_MASK, H_PRIME,
            dO, dMEAN, dRSTD,
            dQ, dK, dV, dGK, dHS,
            bz, nh, l, d_k, d_v, block_size_v, 
            eps, is_rms, timenorm, residual, cummax)
        dQ, dK, dGK = dQ.sum(dim=0), dK.sum(dim=0), dGK.sum(dim=0)
        return dQ, dK, dV, dGK, dHS, None

apply_lam_linear_attention = LamLinearAttention.apply

def lam_linear_attention(q, k, v, gk, num_head, hs=None):
    bz, d_k, d_v = q.shape[0], q.shape[-1], v.shape[-1]
    if hs is None:
        hs = q.new_zeros(bz, num_head, d_v, d_k, dtype=torch.float32)
    he = torch.empty_like(hs, dtype=torch.float32)

    q = rearrange(q, 'b l (n d) -> b n l d', n=num_head)
    k = rearrange(k, 'b l (n d) -> b n l d', n=num_head)
    v = rearrange(v, 'b l (n d) -> b n l d', n=num_head)
    gk = rearrange(gk, 'b l (n d) -> b n l d', n=num_head)
    o, mean, rstd = apply_lam_linear_attention(q, k, v, gk, hs, he)
    o = rearrange(o, 'b n l d -> b l (n d)', n=num_head)
    rstd = rearrange(rstd, 'b n l d -> b l (n d)', n=num_head)
    if mean is not None:
        mean = rearrange(mean, 'b n l d -> b l (n d)', n=num_head)
    return o, he, mean, rstd


if __name__ == "__main__":
    # dtype = torch.bfloat16
    dtype = torch.float32
    device = torch.device("cuda:0")
    bs, l, nh, d_k, d_v = 2, 1024, 4, 32, 64
    q = torch.randn(bs, l, nh * d_k, dtype=dtype, device=device)
    k = torch.randn(bs, l, nh * d_k, dtype=dtype, device=device)
    v = torch.randn(bs, l, nh * d_v, dtype=dtype, device=device)
    g_k = torch.randn(bs, l, nh * d_k, dtype=dtype, device=device)
    hs = torch.randn(bs, nh, d_v, d_k, dtype=dtype, device=device)
    eps = 1e-6
    is_rms = True
    timenorm = True
    residual = False
    cummax = True 
    do = torch.randn(bs, l, nh * d_v, dtype=dtype, device=device)
    # do = torch.zeros_like(do)
    dmean = torch.randn(bs, l, nh * d_v, dtype=dtype, device=device)
    dmean = torch.zeros_like(dmean)
    drstd = torch.randn(bs, l, nh * d_v, dtype=dtype, device=device)
    # drstd = torch.zeros_like(drstd)

    q, k, v, g_k, hs = q.requires_grad_(True), k.requires_grad_(True), v.requires_grad_(True), g_k.requires_grad_(True), hs.requires_grad_(True)

    o1, he1, m1, rstd1 = naive_lam_linear_attention(q, k, v, g_k, hs, nh, eps=eps, is_rms=is_rms, timenorm=timenorm, return_mv=True, residual=residual, cummax=cummax)
    if is_rms:
        final_o1 = o1 * do  + rstd1 * drstd
    else:
        final_o1 = o1 * do + m1 * dmean + rstd1 * drstd
    final_o1.sum().backward()
    # o1.backward(do)
    q_grad_1, k_grad_1, v_grad_1, g_k_grad_1, init_hidden_grad_1 = q.grad.clone(), k.grad.clone(), v.grad.clone(), g_k.grad.clone(), hs.grad.clone()
    q.grad, k.grad, v.grad, g_k.grad, hs.grad = None, None, None, None, None

    o3, he3, m3, rstd3 = lam_linear_attention(q, k, v, g_k, num_head=nh, hs=hs)
    if is_rms:
        final_o3 = o3 * do + rstd3 * drstd
    else:
        final_o3 = o3 * do + m3 * dmean + rstd3 * drstd
    final_o3.sum().backward()
    # o3.backward(do)
    q_grad_3, k_grad_3, v_grad_3, g_k_grad_3, init_hidden_grad_3 = q.grad, k.grad, v.grad, g_k.grad, hs.grad
    print("forward o diff", (o1 - o3).abs().max())
    print("forward he diff", (he1 - he3).abs().max())
    if not is_rms:
        print("forward m diff", (m1 - m3).abs().max())
    print("forward rstd diff", (rstd1 - rstd3).abs().max())
    print("backward q diff", (q_grad_1 - q_grad_3).abs().max())
    print("backward k diff", (k_grad_1 - k_grad_3).abs().max())
    print("backward v diff", (v_grad_1 - v_grad_3).abs().max())
    print("backward gk diff", (g_k_grad_1 - g_k_grad_3).abs().max())
    print("backward hs diff", (init_hidden_grad_1 - init_hidden_grad_3).abs().max())
