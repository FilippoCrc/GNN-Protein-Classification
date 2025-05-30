
import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneralizedCrossEntropy(nn.Module):
    def __init__(self, q=0.7):
        super(GeneralizedCrossEntropy, self).__init__()
        assert q > 0 and q <= 1, "q must be in (0, 1]"
        self.q = q

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
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
    

class GCODLoss(nn.Module):
    """
    Graph Centroid Outlier Discounting (GCOD) Loss Function
    Based on the NCOD method adapted for graph classification.
    The model parameters (theta) are updated using L1 + L3.
    The sample-specific parameters (u) are updated using L2.
    """
    def __init__(self, num_classes=6, alpha_train=0.01, lambda_r=0.1): # Added lambda_r
        """
        Args:
            num_classes (int): Number of classes.
            alpha_train (float): Corresponds to lambda_p in args, coefficient for the
                                 feedback term in L1.
            lambda_r (float): Coefficient for the u regularization term in L2.
        """
        super(GCODLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha_train = alpha_train
        self.lambda_r = lambda_r # Store lambda_r
        self.ce_loss = nn.CrossEntropyLoss(reduction='none') # for per-sample CE

    def _ensure_u_shape(self, u_params, batch_size, target_ndim):
        """Helper to ensure u_params has the correct shape for operations."""
        if u_params.shape[0] != batch_size:
            raise ValueError(f"u_params batch dimension {u_params.shape[0]} does not match expected batch_size {batch_size}")

        if target_ndim == 1: # Expected shape [batch_size]
            return u_params.squeeze() if u_params.ndim > 1 else u_params
        elif target_ndim == 2: # Expected shape [batch_size, 1]
            return u_params.unsqueeze(1) if u_params.ndim == 1 else u_params
        return u_params


    def compute_L1(self, logits, targets, u_params):
        """
        Computes L1 = CE(f_θ(Z_B)) + α_train * u_B * (y_B ⋅ ỹ_B)
        Args:
            logits (Tensor): Model output logits, shape [batch_size, num_classes].
            targets (Tensor): Ground truth labels, shape [batch_size].
            u_params (Tensor): Per-sample u values for the batch, shape [batch_size] or [batch_size, 1].
        Returns:
            Tensor: Scalar L1 loss for the batch.
        """
        batch_size = logits.size(0)
        if batch_size == 0:
            # Corrected line:
            return torch.tensor(0.0, device=logits.device, requires_grad=logits.requires_grad)

        y_onehot = F.one_hot(targets, num_classes=self.num_classes).float()
        y_soft = F.softmax(logits, dim=1)

        ce_loss_values = self.ce_loss(logits, targets) # Shape: [batch_size]

        current_u_params = self._ensure_u_shape(u_params, batch_size, target_ndim=1)

        feedback_term_values = self.alpha_train * current_u_params * (y_onehot * y_soft).sum(dim=1) # Shape: [batch_size]

        L1 = ce_loss_values + feedback_term_values
        return L1.mean()

    def compute_L2(self, logits, targets, u_params):
        """
        Computes L2 = (1/|C|) * ||ỹ_B + u_B * y_B - y_B||²_F + λ_r * ||u_B||²_2
        Args:
            logits (Tensor): Model output logits, shape [batch_size, num_classes].
            targets (Tensor): Ground truth labels, shape [batch_size].
            u_params (Tensor): Per-sample u values for the batch, shape [batch_size] or [batch_size, 1].
        Returns:
            Tensor: Scalar L2 loss for the batch (for u optimization).
        """
        batch_size = logits.size(0)
        if batch_size == 0:
            # Corrected line:
            return torch.tensor(0.0, device=logits.device, requires_grad=logits.requires_grad)

        y_onehot = F.one_hot(targets, num_classes=self.num_classes).float()
        y_soft = F.softmax(logits, dim=1)

        current_u_params_unsqueezed = self._ensure_u_shape(u_params, batch_size, target_ndim=2)

        term = y_soft + current_u_params_unsqueezed * y_onehot - y_onehot # Shape: [batch_size, num_classes]

        # L2 reconstruction term (Frobenius norm for matrix part)
        L2_reconstruction = (1.0 / self.num_classes) * torch.norm(term, p='fro').pow(2)
        
        # u regularization term (L2 norm for u_params vector part)
        # Ensure u_params is 1D for this norm calculation
        current_u_params_1d = self._ensure_u_shape(u_params, batch_size, target_ndim=1)
        u_reg = self.lambda_r * torch.norm(current_u_params_1d, p=2).pow(2)

        L2 = L2_reconstruction + u_reg
        return L2

    def compute_L3(self, logits, targets, u_params, l3_coeff):
        """
        Computes L3 = l3_coeff * D_KL(L || σ(-log(u_B)))
                     where l3_coeff = (1 - training_accuracy)
                     and L = log(σ(logit_true_class)) are log-probabilities
                     and σ(-log(u_B)) are probabilities
        Args:
            logits (Tensor): Model output logits, shape [batch_size, num_classes].
            targets (Tensor): Ground truth labels, shape [batch_size].
            u_params (Tensor): Per-sample u values for the batch, shape [batch_size] or [batch_size, 1].
            l3_coeff (float): Coefficient for the KL divergence term, e.g., (1 - training_accuracy).
        Returns:
            Tensor: Scalar L3 loss for the batch.
        """
        batch_size = logits.size(0)
        if batch_size == 0:
            # Corrected line:
            return torch.tensor(0.0, device=logits.device, requires_grad=logits.requires_grad)

        y_onehot = F.one_hot(targets, num_classes=self.num_classes).float()

        # Logit of the true class for each sample in the batch
        diag_elements = (logits * y_onehot).sum(dim=1) # Shape: [batch_size]

        # L_log_probs = log(sigma(true_class_logit)) which are log-probabilities
        L_log_probs = F.logsigmoid(diag_elements) # Shape: [batch_size]

        current_u_params = self._ensure_u_shape(u_params, batch_size, target_ndim=1)

        # target_probs_for_kl = sigma(-log(u_B)) which are probabilities
        target_probs_for_kl = torch.sigmoid(-torch.log(current_u_params + 1e-8)) # Shape: [batch_size]

        # F.kl_div expects input (L_log_probs) as log-probabilities and target (target_probs_for_kl) as probabilities.
        # reduction='mean' averages the loss over all elements in the batch.
        # log_target=False means target_probs_for_kl are probabilities, not log-probabilities.
        kl_div = F.kl_div(L_log_probs, target_probs_for_kl, reduction='batchmean', log_target=False)

        L3 = l3_coeff * kl_div
        return L3

    def forward(self, logits, targets, u_params, training_accuracy):
        """
        Calculates the GCOD loss components.
        The main loss for model (theta) update is L1 + L3.
        L2 is primarily used for updating u_params (called separately).
        Args:
            logits (Tensor): Model output logits.
            targets (Tensor): Ground truth labels.
            u_params (Tensor): Per-sample u values for the batch.
            training_accuracy (float): The actual training accuracy (value between 0 and 1)
                                     for the current batch or epoch.
        Returns:
            tuple: (total_loss_for_theta, L1, L2, L3)
                   total_loss_for_theta = L1 + L3
        """
        calculated_L1 = self.compute_L1(logits, targets, u_params)
        # L2 is calculated here mainly for complete reporting if needed,
        # but the train loop will call compute_L2 separately for u-optimization.
        # This L2 will now include the regularization term.
        calculated_L2 = self.compute_L2(logits, targets, u_params)

        l3_coefficient = (1.0 - training_accuracy) # As per GCOD paper (1 - alpha_train where alpha_train is accuracy)
        calculated_L3 = self.compute_L3(logits, targets, u_params, l3_coefficient)

        total_loss_for_theta = calculated_L1 + calculated_L3

        return total_loss_for_theta, calculated_L1, calculated_L2, calculated_L3
    
