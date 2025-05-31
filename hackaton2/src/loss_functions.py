import torch
import torch.nn as nn
import torch.nn.functional as F

class GeneralizedCrossEntropy(nn.Module):
    def __init__(self, q=0.7):
        super(GeneralizedCrossEntropy, self).__init__()
        assert q > 0 and q <= 1, "q must be in (0, 1]"
        self.q = q

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        target_probs = probs[torch.arange(targets.size(0)), targets]
        loss = (1 - (target_probs ** self.q)) / self.q
        return loss.mean()
    
class SymmetricCrossEntropy(nn.Module):
    def __init__(self, alpha=0.3, beta=1.0):
        super(SymmetricCrossEntropy, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets)

        probs = F.softmax(logits, dim=1)
        one_hot = F.one_hot(targets, num_classes=logits.size(1)).float()
        rce = -torch.sum(probs * torch.log(one_hot + 1e-7), dim=1).mean()

        loss = self.alpha * ce + self.beta * rce
        return loss

class NoisyCrossEntropyLoss(torch.nn.Module):
    def __init__(self, p_noisy):
        super().__init__()
        self.p = p_noisy
        self.ce = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, targets):
        losses = self.ce(logits, targets)
        weights = (1 - self.p) + self.p * (1 - torch.nn.functional.one_hot(targets, num_classes=logits.size(1)).float().sum(dim=1))
        return (losses * weights).mean()