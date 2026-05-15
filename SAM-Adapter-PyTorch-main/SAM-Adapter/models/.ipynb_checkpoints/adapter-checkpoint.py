import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

class Adapter(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        mlp_ratio: float = 0.25,
        act_layer: nn.Module = nn.GELU,
        skip_connect: bool = True
    ):
        super().__init__()
        self.skip_connect = skip_connect
        self.hidden_features = int(in_features * mlp_ratio)
        self.act = act_layer()
        self.down_proj = nn.Linear(in_features, self.hidden_features)
        self.up_proj = nn.Linear(self.hidden_features, out_features)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_normal_(self.down_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.xavier_normal_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x):
        xs = self.down_proj(x)
        xs = self.act(xs)
        xs = self.up_proj(xs)
        return x + xs if self.skip_connect else xs

    def save_parameters(self, adapter_tensors, prefix):
        adapter_tensors[f"{prefix}.down_proj.weight"] = self.down_proj.weight
        adapter_tensors[f"{prefix}.up_proj.weight"] = self.up_proj.weight
        adapter_tensors[f"{prefix}.down_proj.bias"] = self.down_proj.bias
        adapter_tensors[f"{prefix}.up_proj.bias"] = self.up_proj.bias

    def load_parameters(self, state_dict, prefix):
        self.down_proj.weight = Parameter(state_dict[f"{prefix}.down_proj.weight"])
        self.up_proj.weight = Parameter(state_dict[f"{prefix}.up_proj.weight"])
        self.down_proj.bias = Parameter(state_dict[f"{prefix}.down_proj.bias"])
        self.up_proj.bias = Parameter(state_dict[f"{prefix}.up_proj.bias"])


class _adapter_attn(nn.Module):
    def __init__(
        self,
        block_attn_proj: nn.Module,
        adapter_attn: nn.Module,
    ):
        super().__init__()
        assert isinstance(block_attn_proj, nn.Module) and isinstance(adapter_attn, nn.Module), \
            "block_attn_proj and adapter_attn must be instances of nn.Module"
        self.proj = block_attn_proj
        self.adapter_attn = adapter_attn

    def forward(self, x):
        x = self.proj(x)
        x = self.adapter_attn(x)
        return x


class _adapter_mlp(nn.Module):
    def __init__(
        self,
        block_mlp: nn.Module,
        adapter_mlp: nn.Module,
        index: int = -1,
        scale: float = 0.5,
    ):
        super().__init__()
        assert isinstance(block_mlp, nn.Module) and isinstance(adapter_mlp, nn.Module), \
            "block_mlp and adapter_mlp must be instances of nn.Module"
        self.adapter = lambda x: adapter_mlp(x, index) if index >= 0 else adapter_mlp(x)

        self.scale = scale
        self.lin1 = block_mlp.lin1
        self.lin2 = block_mlp.lin2
        self.act = block_mlp.act
        self.adapter_mlp = adapter_mlp

    def forward(self, x):
        ax = self.adapter(x)
        x = self.lin1(x)
        x = self.act(x)
        x = self.lin2(x)
        return x + ax * self.scale
        # return x