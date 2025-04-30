import torch
import torch.nn as nn
from fp4_torch_kernel.utils import FP4ToBF16Function

class FP4Linear(nn.Module):
    def __init__(self, in_features, out_features, weight_data = None, bias_data = None, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        if weight_data is not None:
            weight = weight_data
        else:
            weight = torch.empty(out_features, in_features, dtype=torch.bfloat16)
        self.weight = nn.Parameter(weight.view(torch.float4_e2m1fn_x2))

        if bias:
            if bias_data is not None:
                bias_tensor = bias_data
            else:
                bias_tensor = torch.empty(out_features, dtype=torch.bfloat16)
            self.bias = nn.Parameter(bias_tensor.view(torch.float4_e2m1fn_x2))
        else:
            self.bias = None
        self.reset_parameters((False if weight_data is not None else True), (False if bias_data is not None else True))

    def reset_parameters(self, reset_weights, reset_bias):
        # Standard initialization (here using normal distribution in bfloat16) then cast to float4.
        if reset_weights:
            weight_bf16 = torch.empty_like(self.weight.data.view(torch.bfloat16)).normal_()
            self.weight.data = weight_bf16.view(torch.float4_e2m1fn_x2)
        if self.bias is not None and reset_bias:
            bias_bf16 = torch.empty_like(self.bias.data.view(torch.bfloat16)).normal_()
            self.bias.data = bias_bf16.view(torch.float4_e2m1fn_x2)

    def forward(self, input):
        weight_bf16 = FP4ToBF16Function.apply(self.weight)
        bias_bf16 = None
        if self.bias is not None:
            bias_bf16 = FP4ToBF16Function.apply(self.bias)
        return nn.functional.linear(input, weight_bf16, bias_bf16)
    
    def named_parameters(self, prefix='', recurse=True):
        # Override to expose parameters as bfloat16 to FSDP, DDP, etc.
        for name, param in super().named_parameters(prefix=prefix, recurse=recurse):
            yield name, param.view(torch.bfloat16)

    def parameters(self, recurse=True):
        for param in super().parameters(recurse=recurse):
            yield param.view(torch.bfloat16)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        # Save parameters as bfloat16
        state_dict = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        for k, v in state_dict.items():
            if v.dtype == torch.float4_e2m1fn_x2:
                state_dict[k] = v.view(torch.bfloat16)
        return state_dict
    
    def load_state_dict(self, state_dict, strict=True):
        # Convert incoming params from bfloat16 (or float32) to float4_e2m1fn_x2
        new_state = {}
        for k, v in state_dict.items():
            if v.dtype in [torch.bfloat16, torch.float32]:
                new_state[k] = v.view(torch.float4_e2m1fn_x2)
            elif v.dtype == torch.float4_e2m1fn_x2:
                new_state[k] = v
            else:
                raise ValueError(f"Unsupported dtype {v.dtype} in checkpoint for key '{k}'")
        super().load_state_dict(new_state, strict=strict)