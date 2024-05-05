import torch 
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from causal_conv1d import causal_conv1d_fn
from fla.ops.gla.recurrent_fuse import fused_recurrent_gla
from fla.ops.gla.chunk import chunk_gla
from fla.modules import RMSNorm

import triton
import triton.language as tl

from torch.cuda.amp import custom_bwd, custom_fwd
from zoology.mixers.lam_scan.lam_scan import apply_lam_linear_attention
from zoology.mixers.lam_scan.lam_1d_scan import apply_1d_lam_linear_attention

import functools

def contiguous(fn):
    @functools.wraps(fn)
    def wrapper(ctx, *args, **kwargs):
        args = (i if not isinstance(i, torch.Tensor) else i.contiguous() for i in args)
        return fn(ctx,
                  *(i if not isinstance(i, torch.Tensor) else i.contiguous() for i in args),
                  **{k: (v if not isinstance(v, torch.Tensor) else v.contiguous()) for k, v in kwargs.items()})
    return wrapper

class RMSLinearAttention(nn.Module):
    def __init__(
            self, 
            d_model: int, 
            num_heads: int, 
            expand_k: float = 1.0,
            expand_v: float = 2.0,
            gate_logit_normalizer: int = 16,
            layer_idx: int=None,
        ) -> None:
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads

        # self.qk_dim = int(d_model * expand_k) + 16 * self.num_heads
        self.qk_dim = int(d_model * expand_k)
        self.v_dim = int(d_model * expand_v)
        self.qk_dim_per_head = self.qk_dim // self.num_heads
        self.v_dim_per_head = self.v_dim // self.num_heads

        self.q_proj = nn.Linear(self.d_model, self.qk_dim, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.qk_dim, bias=False)
        self.k_gate = nn.Linear(self.d_model, self.qk_dim, bias=True) 
        # lowrank_dim = 16
        # self.k_gate = nn.Sequential(nn.Linear(self.d_model, lowrank_dim, bias=False), nn.Linear(lowrank_dim, self.qk_dim, bias=True))
        self.v_proj = nn.Linear(self.d_model, self.v_dim, bias=False)
        self.g_proj = nn.Linear(self.d_model, self.v_dim, bias=True)
        self.out_proj = nn.Linear(self.v_dim, self.d_model, bias=False)

        self.gate_logit_normalizer = gate_logit_normalizer

        self.scaling = self.qk_dim_per_head ** -0.5
        # self.scaling = self.qk_dim_per_head
        # self.gate_fn = nn.SiLU()
        # self.gate_fn = nn.ReLU()
        # self.gate_fn = nn.GELU()
        # self.gate_fn = nn.Sigmoid()

        # self.q_layernorm = nn.LayerNorm(self.qk_dim_per_head, eps=1e-5, elementwise_affine=False, bias=False)
        # self.v_layernorm = RMSNorm(self.v_dim_per_head, eps=1e-5, elementwise_affine=False)

        # 1e-5 is used in the official implementation
        self.group_norm = nn.LayerNorm(self.v_dim_per_head, eps=1e-5, elementwise_affine=False)
        # self.group_norm = RMSNorm(self.v_dim_per_head, eps=1e-5, elementwise_affine=False)

        # d_conv = 4 # following mamba
        # self.q_conv1d = nn.Conv1d(
        #     in_channels=qk_dim,
        #     out_channels=qk_dim,
        #     bias=True,
        #     kernel_size=4,
        #     groups=qk_dim,
        #     padding=d_conv - 1,
        # )
        # self.k_conv1d = nn.Conv1d(
        #     in_channels=qk_dim,
        #     out_channels=qk_dim,
        #     bias=True,
        #     kernel_size=4,
        #     groups=qk_dim,
        #     padding=d_conv - 1,
        # )
        # self.v_conv1d = nn.Conv1d(
        #     in_channels=self.d_model,
        #     out_channels=self.d_model,
        #     bias=True,
        #     kernel_size=4,
        #     groups=self.d_model,
        #     padding=d_conv - 1,
        # )

    def post_init(self):
        # init other than output projection is from retnet
        nn.init.xavier_uniform_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=2 ** -2.5)
                    
        # nn.init.xavier_uniform_(self.k_gate.weight, gain=2 ** -2.5)
        if isinstance(self.k_gate, nn.Sequential):
            nn.init.xavier_uniform_(self.k_gate[0].weight, gain=2 ** -2.5)
            nn.init.xavier_uniform_(self.k_gate[1].weight, gain=2 ** -2.5)
        else:
            nn.init.xavier_uniform_(self.k_gate.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.g_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=2 ** -2.5)
        # nn.init.normal_(self.out_proj.weight, std=0.02 / (2 * self.n_layer))

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x) 
        gk = self.k_gate(x)
        v = self.v_proj(x) 
        
        # q = causal_conv1d_fn(
        #     q.transpose(1, 2), 
        #     weight=rearrange(self.q_conv1d.weight, 'd 1 w -> d w'),
        #     bias=self.q_conv1d.bias, 
        #     activation=None).transpose(1, 2)
        # k = causal_conv1d_fn(
        #     k.transpose(1, 2),
        #     weight=rearrange(self.k_conv1d.weight, 'd 1 w -> d w'),
        #     bias=self.k_conv1d.bias,
        #     activation=None).transpose(1, 2)
        # v = causal_conv1d_fn(
        #     v.transpose(1, 2),
        #     weight=rearrange(self.v_conv1d.weight, 'd 1 w -> d w'),
        #     bias=self.v_conv1d.bias,
        #     activation=None).transpose(1, 2)
        
        # k = k * self.scaling
        # g = self.g_proj(x)

        # output = self.naive_rms_linear_attention(q, k, v, gk, num_head=self.num_heads, use_nonlinear=self.use_nonlinear)
        # output = self.naive_rms_linear_attention_1d(q, k, v, gk, num_head=self.num_heads)
        # output, _ = self.rms_linear_attention(q, k, v, gk, num_head=self.num_heads)
        # output = self.naive_rms_linear_attention_timenorm(q, k, v, gk, num_head=self.num_heads)
        # output = self.gated_linear_attention(q, k, v, gk, num_head=self.num_heads)
        # output = self.hybrid_linear_attention(q, k, v, gk, num_head=self.num_heads)
        output = self.rms_1d_linear_attention(k, gk, num_head=self.num_heads)

        # output = self.gate_fn(g) * output
        output = self.out_proj(output)
        return output

    def naive_rms_linear_attention(self, q, k, v, gk, num_head):
        q = rearrange(q, 'b l (n d) -> b n l d', n=num_head)
        k = rearrange(k, 'b l (n d) -> b n l d', n=num_head)
        v = rearrange(v, 'b l (n d) -> b n l d', n=num_head)
        gk = rearrange(gk, 'b l (n d) -> b n l d', n=num_head)
        # gk = gk.sigmoid() # b x n x l x d
        gk = torch.ones_like(gk)

        b, n, l, d_k = q.shape
        d_v = v.shape[-1]
        h = torch.zeros((b, n, d_v, d_k), dtype=q.dtype, device=q.device)
        o_l = []
        h_square_history = None
        for i in range(l):
            q_i = q[:, :, i] # b x n x d_k
            k_i = k[:, :, i] # b x n x d_k
            v_i = v[:, :, i] # b x n x d_v
            gk_i = gk[:, :, i] # b x n x d_k

            rms_h = (h ** 2).norm(dim=-2, keepdim=True) * d_k ** (-1. / 2) # b x n x 1 x d_v
            h_hat = h / (rms_h + 1e-5)

            new_h = h_hat * gk_i[:, :, None, :] + k_i[:, :, None, :] * v_i[:, :, :, None] # b x n x d_v x d_k

            o_i = (q_i[:, :, None, :] * new_h).sum(-1) # b x n x d_v
            o_l.append(o_i)
            h = new_h
        o = torch.stack(o_l, dim=2) # b x n x l x d_v

        o = self.group_norm(o)
        o = rearrange(o, 'b n l d -> b l (n d)')
        return o
    
    def naive_rms_linear_attention_timenorm(self, q, k, v, gk, num_head):
        q = rearrange(q, 'b l (n d) -> b n l d', n=num_head)
        k = rearrange(k, 'b l (n d) -> b n l d', n=num_head)
        v = rearrange(v, 'b l (n d) -> b n l d', n=num_head)
        gk = rearrange(gk, 'b l (n d) -> b n l d', n=num_head)

        # gk = gk.sigmoid() # b x n x l x d
        # gk = torch.ones_like(gk)
        q = q * self.scaling

        b, n, l, d_k = q.shape
        d_v = v.shape[-1]
        h = torch.zeros((b, n, d_v, d_k), dtype=q.dtype, device=q.device)
        o_l = []
        h_square_history = None
        for i in range(0, l): 
            q_i = q[:, :, i] # b x n x d_k
            k_i = k[:, :, i] # b x n x d_k
            v_i = v[:, :, i] # b x n x d_v
            gk_i = gk[:, :, i] # b x n x d_k

            if i == 0:
                h_hat = 0
                new_h = k_i[:, :, None, :] * v_i[:, :, :, None]

                h_square_history = new_h ** 2
            else:
                _n_channel = i * d_k # i is zero-based
                rms_h_square = h_square_history.sum(dim=-1, keepdim=True) / _n_channel
                rms_h = torch.sqrt(rms_h_square)
                h_hat = h / (rms_h + 1e-5)
                new_h = h_hat * gk_i[:, :, None, :] + k_i[:, :, None, :] * v_i[:, :, :, None]

                h_square_history = h_square_history + new_h ** 2

            new_h = h_hat * gk_i[:, :, None, :] + k_i[:, :, None, :] * v_i[:, :, :, None] # b x n x d_v x d_k

            o_i = (q_i[:, :, None, :] * new_h).sum(-1) # b x n x d_v
            o_l.append(o_i)
            h = new_h
        o = torch.stack(o_l, dim=2) # b x n x l x d_v

        # o = self.group_norm(o)
        o = rearrange(o, 'b n l d -> b l (n d)')
        return o

    def naive_rms_linear_attention_1d(self, q, k, v, gk, num_head=8):
        # h = g \odot h + k \odot v
        q = rearrange(q, 'b l (n d) -> b n l d', n=num_head)
        k = rearrange(k, 'b l (n d) -> b n l d', n=num_head)
        v = rearrange(v, 'b l (n d) -> b n l d', n=num_head)
        gk = rearrange(gk, 'b l (n d) -> b n l d', n=num_head)
        
        k = k.sigmoid()
        gk = (1 - k)

        b, n, l, d_k = q.shape
        d_v = v.shape[-1]

        assert d_k == d_v
        h = torch.zeros((b, n, d_v), dtype=q.dtype, device=q.device)
        o_l = []
        for i in range(l):
            # q_i = q[:, :, i] # b x n x d_k
            k_i = k[:, :, i] # b x n x d_k
            v_i = v[:, :, i] # b x n x d_v
            gk_i = gk[:, :, i] # b x n x d_k

            rms_h = (h ** 2).norm(dim=-1, keepdim=True) * d_k ** (-1. / 2) # b x n x 1 x d_v
            h_hat = h / (rms_h + 1e-5)

            new_h = h_hat * gk_i + k_i * v_i # b x n x d_v
            o_l.append(new_h)

        o = torch.stack(o_l, dim=2) # b x n x l x d_v

        o = self.group_norm(o)
        o = rearrange(o, 'b n l d -> b l (n d)')
        return o

    def rms_linear_attention(self, q, k, v, gk, num_head, hs=None):
        bz, d_k, d_v = q.shape[0], q.shape[-1], v.shape[-1]
        if hs is None:
            hs = q.new_zeros(bz, num_head, d_v, d_k, dtype=torch.float32)
        he = torch.empty_like(hs, dtype=torch.float32)

        q = rearrange(q, 'b l (n d) -> b n l d', n=num_head)
        k = rearrange(k, 'b l (n d) -> b n l d', n=num_head)
        v = rearrange(v, 'b l (n d) -> b n l d', n=num_head)
        gk = rearrange(gk, 'b l (n d) -> b n l d', n=num_head)

        # q = self.q_layernorm(q)
        # q = q / q.norm(dim=-1, keepdim=True)
        # q = self.gate_fn(q)
        q = F.relu(q)
        k = F.relu(k)
        # q = F.elu(q) + 1.0
        # k = F.elu(k) + 1.0
        # k = self.gate_fn(k)
        # k = F.sigmoid(k)
        # v = F.silu(v)
        # v = self.v_layernorm(v)
        # v = F.tanh(v)
        q = q * self.scaling
        # q = q * self.scaling ** 2
        v = F.silu(v)
        # v = F.tanh(v)

        # gk = F.relu(gk)
        # gk = gk.sigmoid()
        # gk = F.silu(gk)
        # gk = torch.exp(-torch.exp(gk / self.gate_logit_normalizer))
        # gk = self.gate_fn(gk)
        # gk = (F.logsigmoid(gk) / 16.).exp() + 1.0
        # gk = (F.logsigmoid(gk) / 16.).exp()
        gk = (F.logsigmoid(gk) / self.gate_logit_normalizer).exp()
        # gk = (F.logsigmoid(gk) / self.gate_logit_normalizer).exp() * 8
        # gk = (F.logsigmoid(gk) / self.gate_logit_normalizer).exp() * 2 - 1.0
        # gk = (F.logsigmoid(gk) / self.gate_logit_normalizer).exp() + 1.0
        # gk = (F.logsigmoid(gk) / 64.).exp()
        # gk = torch.ones_like(gk)
        # k = k.sigmoid()
        # gk = 1 - k

        o, _, _ = apply_lam_linear_attention(q, k, v, gk, hs, he)
        # o = self.group_norm(o)
        o = rearrange(o, 'b n l d -> b l (n d)', n=num_head)
        return o, he

    def gated_linear_attention(self, q, k, v, gk, num_head):
        # basically without the rms part
        q = rearrange(q, 'b l (n d) -> b n l d', n=num_head).contiguous()
        k = rearrange(k, 'b l (n d) -> b n l d', n=num_head).contiguous()
        v = rearrange(v, 'b l (n d) -> b n l d', n=num_head).contiguous()
        gk = rearrange(gk, 'b l (n d) -> b n l d', n=num_head).to(torch.float32).contiguous()
        gk = F.logsigmoid(gk) / self.gate_logit_normalizer

        # # q, k = F.relu(q), F.relu(k)
        # o, _ = fused_recurrent_gla(q, k, v, gk)
        o, _ = chunk_gla(q, k, v, gk)
        o = self.group_norm(o)
        o = rearrange(o, 'b n l d -> b l (n d)', n=num_head)
        return o

    def hybrid_linear_attention(self, q, k, v, gk, num_head, hs=None):
        bz, d_k_, d_v = q.shape[0], q.shape[-1], v.shape[-1]
        d_k_1, d_k_2 = 32, d_k_ - 32

        # lam 
        if hs is None:
            hs = q.new_zeros(bz, num_head, d_v, d_k_1, dtype=torch.float32)
        he = torch.empty_like(hs, dtype=torch.float32)

        q = rearrange(q, 'b l (n d) -> b n l d', n=num_head)
        k = rearrange(k, 'b l (n d) -> b n l d', n=num_head)
        v = rearrange(v, 'b l (n d) -> b n l d', n=num_head)
        gk = rearrange(gk, 'b l (n d) -> b n l d', n=num_head)

        q = F.relu(q)
        k = F.relu(k)
        v = F.silu(v)
        gk = F.logsigmoid(gk) / self.gate_logit_normalizer

        q1 = q[:, :, :, :d_k_1]
        q2 = q[:, :, :, d_k_1:]
        k1 = k[:, :, :, :d_k_1]
        k2 = k[:, :, :, d_k_1:]
        gk1 = gk[:, :, :, :d_k_1]
        gk2 = gk[:, :, :, d_k_1:]
        
        q1 = q1 * self.scaling
        gk1 = gk1.exp()
        o1, mean, rstd = apply_lam_linear_attention(q1, k1, v, gk1, hs, he)
        # rstd = rstd.detach()

        gv2 = torch.log(rstd)
        o2, _ = fused_recurrent_gla(q2, k2, v, gk2, gv2, scale=self.scaling)
        o = o1 + o2
        # o = self.group_norm(o)
        o = rearrange(o, 'b n l d -> b l (n d)', n=num_head)
        return o
    
    def rms_1d_linear_attention(self, x, g, num_head):
        g = g.sigmoid()
        bz, d = x.shape[0], x.shape[-1]
        hs = x.new_zeros(bz, num_head, d, dtype=torch.float32)

        x = rearrange(x, 'b l (n d) -> b n l d', n=num_head)
        g = rearrange(g, 'b l (n d) -> b n l d', n=num_head)
        h = apply_1d_lam_linear_attention(x, g, hs)
        h = rearrange(h, 'b n l d -> b l (n d)', n=num_head)
        return h

    def state_size(self, batch_size: int=1, sequence_length: int=2048):
        return self.d_model ** 2 / self.num_heads