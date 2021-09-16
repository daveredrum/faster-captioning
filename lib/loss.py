import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftmaxRankingLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):
        # input check
        assert inputs.shape == targets.shape
        
        # compute the probabilities
        probs = F.softmax(inputs + 1e-8, dim=1)

        # reduction
        loss = -torch.sum(torch.log(probs + 1e-8) * targets, dim=1).mean()

        return loss

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.2, gamma=5):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.gamma = gamma

    def forward(self, score, label):
        score *= self.gamma
        sim = (score*label).sum()
        neg_sim = score*label.logical_not()
        neg_sim = torch.logsumexp(neg_sim, dim=0) # soft max
        loss = torch.clamp(neg_sim - sim + self.margin, min=0).sum()

        return loss