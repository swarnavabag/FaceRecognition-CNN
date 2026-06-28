import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# ARCFACE MARGIN LAYER
# ============================================================

class ArcMarginProduct(nn.Module):
    """
    ArcFace layer.

    Applies additive angular margin penalty:
        cos(theta + m)

    Parameters:
        in_features  : embedding dimension (512)
        out_features : number of identities/classes
        s            : scale factor
        m            : angular margin
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        s: float = 30.0,
        m: float = 0.3
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m

        self.weight = nn.Parameter(
            torch.FloatTensor(out_features, in_features)
        )

        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)

    def forward(self, embeddings, labels):
        """
        embeddings: [batch_size, 512]
        labels:     [batch_size]
        """

        # Normalize embeddings and weights
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weights = F.normalize(self.weight, p=2, dim=1)

        # Cosine similarity
        cosine = F.linear(embeddings, weights)

        # Numerical safety
        cosine = torch.clamp(cosine, -1.0, 1.0)

        # Compute sine(theta)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))

        # cos(theta + m)
        phi = cosine * self.cos_m - sine * self.sin_m

        # One-hot labels
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)

        # Apply margin only to true class
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)

        # Scale logits
        output *= self.s

        return output