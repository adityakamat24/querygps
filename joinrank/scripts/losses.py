#!/usr/bin/env python3
"""
Loss functions for join order ranking
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class PairwiseHingeLoss(nn.Module):
    """
    Pairwise hinge loss for ranking.
    For each pair of orders, encourage the model to rank the better one higher.
    """
    
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin
    
    def forward(self, pred_scores, true_scores):
        """
        Args:
            pred_scores: [batch_size, num_orders] predicted scores
            true_scores: [batch_size, num_orders] true relevance scores
        Returns:
            loss: scalar
        """
        batch_size, num_orders = pred_scores.shape
        
        # Generate all pairwise comparisons
        loss = 0.0
        count = 0

        for i in range(num_orders):
            for j in range(i + 1, num_orders):
                # Compare scores for each example in the batch
                i_better = true_scores[:, i] > true_scores[:, j]  # [batch_size]
                j_better = true_scores[:, j] > true_scores[:, i]  # [batch_size]

                # If order i is better than order j (higher true score)
                if i_better.any():
                    # We want pred_scores[:, i] > pred_scores[:, j] + margin
                    pair_loss = F.relu(
                        self.margin - (pred_scores[:, i] - pred_scores[:, j])
                    )
                    # Only apply loss where i is actually better
                    masked_loss = pair_loss * i_better.float()
                    loss += masked_loss.sum()
                    count += i_better.sum().item()

                # If order j is better than order i
                if j_better.any():
                    pair_loss = F.relu(
                        self.margin - (pred_scores[:, j] - pred_scores[:, i])
                    )
                    # Only apply loss where j is actually better
                    masked_loss = pair_loss * j_better.float()
                    loss += masked_loss.sum()
                    count += j_better.sum().item()
        
        if count > 0:
            loss = loss / count
        
        return loss

class ListwiseLoss(nn.Module):
    """
    Listwise loss using negative log-likelihood of ranking.
    Based on ListNet / ListMLE approaches.
    """
    
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, pred_scores, true_scores):
        """
        Args:
            pred_scores: [batch_size, num_orders]
            true_scores: [batch_size, num_orders]
        Returns:
            loss: scalar
        """
        # Apply softmax to get probability distributions
        pred_probs = F.softmax(pred_scores / self.temperature, dim=1)
        true_probs = F.softmax(true_scores / self.temperature, dim=1)
        
        # KL divergence between true and predicted distributions
        loss = F.kl_div(
            pred_probs.log(),
            true_probs,
            reduction='batchmean'
        )
        
        return loss

class ApproxNDCGLoss(nn.Module):
    """
    Differentiable approximation of NDCG loss.
    Based on "Learning to Rank with Nonsmooth Cost Functions" (Burges et al.)
    """
    
    def __init__(self, temperature: float = 1.0, k: int = 10):
        super().__init__()
        self.temperature = temperature
        self.k = k
    
    def forward(self, pred_scores, true_scores):
        """
        Args:
            pred_scores: [batch_size, num_orders]
            true_scores: [batch_size, num_orders]
        Returns:
            loss: scalar (negative NDCG to minimize)
        """
        batch_size = pred_scores.size(0)
        k = min(self.k, pred_scores.size(1))
        
        # Compute gains (use true scores as relevance)
        gains = true_scores
        
        # Compute DCG using soft top-k
        # Use softmax as differentiable approximation of ranking
        soft_ranks = F.softmax(pred_scores / self.temperature, dim=1)
        
        dcg = 0.0
        for i in range(k):
            # Discount factor: 1 / log2(i + 2)
            discount = 1.0 / torch.log2(torch.tensor(i + 2.0))
            dcg += (soft_ranks * gains).sum(dim=1) * discount
        
        # Compute IDCG (ideal DCG)
        sorted_gains, _ = torch.sort(gains, dim=1, descending=True)
        idcg = 0.0
        for i in range(k):
            discount = 1.0 / torch.log2(torch.tensor(i + 2.0))
            idcg += sorted_gains[:, i] * discount
        
        # Normalize
        ndcg = dcg / (idcg + 1e-10)
        
        # Return negative NDCG as loss (to minimize)
        return -ndcg.mean()

class LambdaRankLoss(nn.Module):
    """
    LambdaRank-style loss with gradient based on ΔAP or ΔNDCG.
    Simplified version for efficiency.
    """
    
    def __init__(self, sigma: float = 1.0):
        super().__init__()
        self.sigma = sigma
    
    def forward(self, pred_scores, true_scores):
        """
        Args:
            pred_scores: [batch_size, num_orders]
            true_scores: [batch_size, num_orders]
        Returns:
            loss: scalar
        """
        batch_size, num_orders = pred_scores.shape
        
        loss = 0.0
        count = 0
        
        for i in range(num_orders):
            for j in range(i + 1, num_orders):
                # Skip if scores are equal
                if torch.abs(true_scores[:, i] - true_scores[:, j]) < 1e-6:
                    continue
                
                # Determine which is better
                better_idx = i if true_scores[:, i] > true_scores[:, j] else j
                worse_idx = j if better_idx == i else i
                
                # Compute pairwise loss with sigmoid
                s_ij = pred_scores[:, better_idx] - pred_scores[:, worse_idx]
                pair_loss = torch.log(1 + torch.exp(-self.sigma * s_ij))
                
                # Weight by NDCG gain (simplified)
                delta_gain = torch.abs(true_scores[:, i] - true_scores[:, j])
                loss += (pair_loss * delta_gain).mean()
                count += 1
        
        if count > 0:
            loss = loss / count
        
        return loss

# Example usage and comparison
if __name__ == "__main__":
    # Simulate predictions and targets
    batch_size = 8
    num_orders = 16
    
    pred_scores = torch.randn(batch_size, num_orders)
    true_runtimes = torch.rand(batch_size, num_orders) * 1000 + 100
    true_scores = -torch.log(true_runtimes)  # Higher score = lower runtime
    
    # Test different losses
    print("Testing loss functions:\n")
    
    hinge_loss = PairwiseHingeLoss(margin=1.0)
    loss_val = hinge_loss(pred_scores, true_scores)
    print(f"Pairwise Hinge Loss: {loss_val.item():.4f}")
    
    listwise_loss = ListwiseLoss(temperature=1.0)
    loss_val = listwise_loss(pred_scores, true_scores)
    print(f"Listwise Loss: {loss_val.item():.4f}")
    
    ndcg_loss = ApproxNDCGLoss(temperature=1.0, k=10)
    loss_val = ndcg_loss(pred_scores, true_scores)
    print(f"Approx NDCG Loss: {loss_val.item():.4f}")
    
    lambda_loss = LambdaRankLoss(sigma=1.0)
    loss_val = lambda_loss(pred_scores, true_scores)
    print(f"LambdaRank Loss: {loss_val.item():.4f}")