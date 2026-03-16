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
  
class TripletLoss(nn.Module):
    """Works directly with embeddings rather than with positives, negatives and anchors."""
    def __init__(self, margin=0.3, mining="hard", norm_feat=True, debug=False):
        super().__init__()
        self.margin = margin
        self.mining = mining.lower()
        self.norm_feat = norm_feat # If true, uses Cosine, else Euclidean
        self.debug = debug 
        
    def softmax_weights(self, dist, mask):
        max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
        diff = dist - max_v
        Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6
        return torch.exp(diff) * mask / Z

    def forward(self, embedding, targets):
        # Distance Matrix
        if self.norm_feat:
            # Cosine distance
            embedding = F.normalize(embedding, p=2, dim=1)
            dist_mat = 1 - torch.mm(embedding, embedding.t())
        else:
            # Euclidean distance
            dist_mat = torch.cdist(embedding, embedding, p=2)

        N = dist_mat.size(0)
        # Masks
        is_pos = targets.view(N, 1).expand(N, N).eq(targets.view(N, 1).expand(N, N).t()).float()
        is_neg = targets.view(N, 1).expand(N, N).ne(targets.view(N, 1).expand(N, N).t()).float()
        
        # Remove self-matching
        is_pos = is_pos - torch.eye(N, device=dist_mat.device)

        INF = torch.finfo(dist_mat.dtype).max
        # Mining
        if self.mining == "hard":
            dist_ap = dist_mat.masked_fill(is_pos == 0, -INF).max(dim=1)[0]
            dist_an = dist_mat.masked_fill(is_neg == 0,  INF).min(dim=1)[0]
        elif self.mining == "weighted":
            # Weighted Positive
            w_ap = self.softmax_weights(dist_mat * is_pos, is_pos)
            dist_ap = torch.sum(dist_mat * is_pos * w_ap, dim=1)
            # Weighted Negative
            w_an = self.softmax_weights(-dist_mat * is_neg, is_neg)
            dist_an = torch.sum(dist_mat * is_neg * w_an, dim=1)
        elif self.mining == "random":
            # Randomly select one positive and one negative per anchor
            dist_ap = torch.zeros(N, device=dist_mat.device)
            dist_an = torch.zeros(N, device=dist_mat.device)
            for i in range(N):
                pos_idx = torch.where(is_pos[i] > 0)[0]
                neg_idx = torch.where(is_neg[i] > 0)[0]
                if len(pos_idx) > 0:
                    dist_ap[i] = dist_mat[i, pos_idx[torch.randint(len(pos_idx), (1,))]]
                else:
                    dist_ap[i] = 0.0  # fallback if no positive (should not happen)
                if len(neg_idx) > 0:
                    dist_an[i] = dist_mat[i, neg_idx[torch.randint(len(neg_idx), (1,))]]
                else:
                    dist_an[i] = INF  # fallback if no negative (should not happen)
        else: # semi-hard fallback
            dist_ap = dist_mat.masked_fill(is_pos == 0, -INF).max(dim=1)[0]
            mask_semi = is_neg * (dist_mat > dist_ap.unsqueeze(1)) * (dist_mat < dist_ap.unsqueeze(1) + self.margin)
            if mask_semi.sum() > 0:
                dist_an = (dist_mat * mask_semi).sum(1) / (mask_semi.sum(1) + 1e-6)
            else:
                dist_an = dist_mat.masked_fill(is_neg == 0,  INF).min(dim=1)[0]

        # Loss calculation
        y = torch.ones_like(dist_an)
        
        if self.debug:
            pos_mean = dist_ap.mean().item()
            neg_mean = dist_an.mean().item()
            violation_rate = (dist_an < dist_ap + self.margin).float().mean().item()

            print(
                f"pos={pos_mean:.3f} "
                f"neg={neg_mean:.3f} "
                f"viol={violation_rate:.3f}"
            )
        
        if self.margin > 0:
            # Standard Triplet (F.softplus or margin_ranking)
            # return F.softplus(dist_ap - dist_an + self.margin)
            return F.margin_ranking_loss(dist_an, dist_ap, y, margin=self.margin)
        else:
            # Soft-margin (no fixed boundary)
            return F.soft_margin_loss(dist_an - dist_ap, y)