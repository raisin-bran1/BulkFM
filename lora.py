import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    def __init__(self, original_layer, rank=8, alpha=16):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        # Freeze original layer
        self.original_layer.weight.requires_grad = False
        if self.original_layer.bias is not None:
            self.original_layer.bias.requires_grad = False
            
    def forward(self, x):
        original_output = self.original_layer(x)
        lora_output = (x @ self.lora_A.t() @ self.lora_B.t()) * self.scaling
        return original_output + lora_output

def apply_lora(model, rank=8, alpha=16):
    """
    Applies LoRA to q, k, v maps in attention and U, V maps in FFN.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Target attention and FFN layers
            if any(target in name for target in ['q_map', 'k_map', 'v_map', 'U_map', 'V_map']):
                # Find the parent module and the attribute name
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                parent = model.get_submodule(parent_name)
                
                # Replace with LoRALinear
                setattr(parent, child_name, LoRALinear(module, rank=rank, alpha=alpha))
    return model

def get_lora_params(model):
    lora_params = []
    for name, param in model.named_parameters():
        if 'lora_' in name:
            lora_params.append(param)
    return lora_params
