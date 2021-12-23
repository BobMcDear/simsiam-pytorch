from typing import Tuple

from torch import randn, Tensor
from torch.nn import BatchNorm1d, Identity, Linear, Module, ReLU, Sequential


def get_encoder_out_dim(encoder: Module) -> int:
    """
    Returns the number of output channels of encoder

    Args:
        encoder (Module): Encoder
    """
    x = randn(1, 3, 128, 128)
    y = encoder(x)
    return y.shape[1]


class LinBnReLU(Module):
    """
    Linear layer followed optionally by batch normalization and ReLU

    Args:
        in_dim (int): Number of input features
        out_dim (int): Number of output features
        bn (bool): Whether to have batch normalization
        relu (bool): Whether to have ReLU
    """
    def __init__(self, in_dim: int, out_dim: int, bn: bool = True, relu: bool = True):
        super().__init__()

        self.linear = Linear(in_dim, out_dim, bias=not bn)
        self.bn = BatchNorm1d(out_dim) if bn else Identity()
        self.relu = ReLU() if relu else Identity()
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.linear(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class SimSiamModel(Module):
    """
    SimSiam model with given encoder

    Args:
        encoder (Module): Encoder
        out_dim (int): Number of output features
        prediction_head_hidden_dim (int): Number of hidden features of prediction head
    """
    def __init__(self, encoder: Module, out_dim: int = 2048, 
                 prediction_head_hidden_dim: int = 512):
        super().__init__()
        
        self.encoder = encoder
        encoder_out_dim = get_encoder_out_dim(self.encoder)

        self.projection_head = Sequential(LinBnReLU(encoder_out_dim, 
                                                    encoder_out_dim),
                                          LinBnReLU(encoder_out_dim, 
                                                    encoder_out_dim),
                                          LinBnReLU(encoder_out_dim, 
                                                    out_dim, relu=False))
        
        self.prediction_head = Sequential(LinBnReLU(out_dim, 
                                                    prediction_head_hidden_dim),
                                          LinBnReLU(prediction_head_hidden_dim,
                                                    out_dim, bn=False, 
                                                    relu=False))
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.encoder(x)
        z = self.projection_head(x)
        p = self.prediction_head(z)
        return z, p
