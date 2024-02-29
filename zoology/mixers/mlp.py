from torch import nn 
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(
        self,
        d_model: int,
        hidden_mult: int=1,
        activation: callable=F.gelu,
        return_residual: bool=False,
        **kwargs
    ):
        super().__init__()
        in_features, out_features = d_model, d_model
        hidden_features = d_model * hidden_mult
        self.return_residual = return_residual
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.activation = activation
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        y = self.fc1(x)
        y = self.activation(y)
        y = self.fc2(y)
        return y if not self.return_residual else (y, x)

class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        intermediate_size: int,
        activation: callable=F.silu,
        return_residual: bool=False,
        **kwargs
    ):
        super().__init__()
        self.hidden_size = d_model
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = activation
        assert not return_residual

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj