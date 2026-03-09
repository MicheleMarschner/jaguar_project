import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ArcFaceLoss(nn.Module):
    """
    Additive Angular Margin Loss (ArcFace)
    Paper: https://arxiv.org/abs/1801.07698
    """
    def __init__(self, s:float = 30.0, m:float = 0.50):
        """
        Args:
            m: multiplicative margin 
            s: scalar scale 
        """
        super().__init__()
        self.s = s
        self.m = m
        # Guard for theta + m > pi
        self.threshold = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, cosine_logits: torch.Tensor, labels: torch.Tensor):
        """
        cosine_logits: (B, C) cosine similarity between features and class weights
        labels: (B,)
        """
        one_hot = F.one_hot(labels, num_classes=cosine_logits.size(1)).float()

        # clamp for numerical stability
        cosine = cosine_logits.clamp(-1 + 1e-7, 1 - 1e-7)
        theta = torch.acos(cosine)
        target_logits = torch.cos(theta + self.m)
        
        # Stability: if theta > pi - m, use a linear sub-optimal margin
        target_logits = torch.where(cosine > self.threshold, target_logits, cosine - self.mm)

        logits = cosine * (1 - one_hot) + target_logits * one_hot
        logits = logits * self.s
        return F.cross_entropy(logits, labels), logits

class CosFaceLoss(nn.Module):
    """
    Additive Margin Softmax (CosFace): differently from Arcface, 
    margin is subtracted directly from cosine logits, not angles.
    """
    def __init__(self, s:float = 30.0, m:float = 0.35):
        """
        Args:
            m: multiplicative margin 
            s: scalar scale 
        """
        super().__init__()
        self.s = s
        self.m = m

    def forward(self, cosine_logits: torch.Tensor, labels: torch.Tensor):
        one_hot = F.one_hot(labels, num_classes=cosine_logits.size(1)).float()
        logits = cosine_logits - one_hot * self.m
        logits = logits * self.s
        return F.cross_entropy(logits, labels), logits

class SphereFaceLoss(nn.Module):
    """
    Multiplicative Angular Margin (SphereFace)
    """
    def __init__(self, s:float=30.0, m:int=4):
        """
        Args:
            m: multiplicative margin 
            s: scalar scale (SphereFace usually uses ||x||, but modern ReID 
               implementations use a fixed scale s for better convergence)
        """
        super().__init__()
        self.s = s
        self.m = m
        
        # Precompute coefficients for cos(m*theta) for m=4
        # cos(4theta) = 8*cos^4(theta) - 8*cos^2(theta) + 1
        self.mlambda = [
            lambda x: x**0,
            lambda x: x**1,
            lambda x: 2*x**2 - 1,
            lambda x: 4*x**3 - 3*x,
            lambda x: 8*x**4 - 8*x**2 + 1,
        ]

    def forward(self, cosine_logits: torch.Tensor, labels: torch.Tensor):
        one_hot = F.one_hot(labels, num_classes=cosine_logits.size(1)).float()
        cosine = cosine_logits.clamp(-1 + 1e-7, 1 - 1e-7)
        
        cos_m_theta = self.mlambda[self.m](cosine)
        # phi(theta) = (-1)^k * cos(m*theta) - 2k
        theta = cosine.acos()
        k = (self.m * theta / math.pi).floor()
        phi_theta = ((-1.0)**k) * cos_m_theta - 2 * k
        
        # Apply margin only to the target (positive) class
        # Replace the target logit with phi_theta
        logits = (one_hot * phi_theta) + ((1.0 - one_hot) * cosine)
        logits = logits * self.s
        return F.cross_entropy(logits, labels), logits

# class TripletLoss(nn.Module):
#     """
#     Standard Triplet Loss. 
#     Note: Requires (Anchor, Positive, Negative) inputs.
#     """
#     def __init__(self, m:float = 0.3):
#         super().__init__()
#         self.m = m

#     def forward(self, anchor, positive, negative):
#         distance = F.triplet_margin_loss(anchor, positive, negative, margin=self.m, p=2)
#         return distance
    
class TripletLoss(nn.Module):
    """
    Batch-hard Triplet Loss for ReID, robust to small numbers of positives.
    Works directly with (embeddings, labels) instead of explicit anchor/pos/neg.
    """
    def __init__(self, m: float = 0.3):
        super().__init__()
        self.margin = m

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor):
        """
        embeddings: (B, D) normalized feature vectors
        labels: (B,) integer class labels
        """
        # pairwise Euclidean distance
        dist_matrix = torch.cdist(embeddings, embeddings, p=2)

        # masks for positives/negatives
        labels = labels.unsqueeze(1)
        mask_pos = labels.eq(labels.t())   # (B,B) same identity
        mask_neg = ~mask_pos               # (B,B) different identity; we don't really need the neg one with the max_dist trick

        # remove self-comparisons
        mask_pos.fill_diagonal_(False)

        # if no positives in batch, return zero loss
        if mask_pos.sum() == 0:
            return torch.tensor(0.0, device=embeddings.device)

        # hardest positive distance for each anchor
        hardest_pos = (dist_matrix * mask_pos.float()).max(dim=1)[0]

        # hardest negative distance for each anchor
        max_dist = dist_matrix.max().detach()
        dist_neg = dist_matrix + max_dist * mask_pos.float()
        hardest_neg = dist_neg.min(dim=1)[0]

        # compute triplet loss
        loss = F.relu(hardest_pos - hardest_neg + self.margin)
        return loss.mean()