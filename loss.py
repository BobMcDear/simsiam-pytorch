from torch import Tensor
from torch.nn import Module
from torch.nn.functional import cosine_similarity


class NegativeCosineSimilarity(Module):
    """
    Negative cosine similarity
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, y1: Tensor, y2: Tensor) -> Tensor:
        loss = -cosine_similarity(y1, y2, dim=1)
        return loss


class SymmetrizedNegativeCosineSimilarity(Module):
    """
    Symmetrized negative cosine similarity loss
    """
    def __init__(self):
        super().__init__()

        self.negative_cosine_similarity = NegativeCosineSimilarity()
    
    def forward(self, p1: Tensor, z1: Tensor, p2: Tensor, 
                z2: Tensor) -> Tensor:
        loss1 = self.negative_cosine_similarity(p1, z2.detach()).mean()
        loss2 = self.negative_cosine_similarity(p2, z1.detach()).mean()
        
        avg_loss = 0.5*loss1 + 0.5*loss2
        return avg_loss
