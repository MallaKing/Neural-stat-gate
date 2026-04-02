import torch
import torch.nn as nn

class NeuralStatisticalGate(nn.Module):
    """
    A Differentiable Statistical Filter utilizing Higher-Order Moments.
    
    ENVIRONMENTAL CONSTRAINT: 
    This model learns a specific statistical mapping (Stats -> Gain) optimized 
    for the specific noise profile (channel environment) it is trained on. 
    Deployment in a novel environment requires fine-tuning the affine parameters.
    """
    def __init__(self, window_size: int = 50, hidden_dim: int = 32):
        super().__init__()
        self.window_size = window_size
        
        self.stats_predictor = nn.Sequential(
            nn.Linear(4, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # Outputs: Gamma (Scale) and Beta (Shift)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        windows = x.unfold(1, self.window_size, 1)
        
        mean = windows.mean(dim=-1, keepdim=True)
        std = windows.std(dim=-1, keepdim=True) + 1e-6
        centered = windows - mean
        
        skew = (centered ** 3).mean(dim=-1, keepdim=True) / (std ** 3)
        kurt = (centered ** 4).mean(dim=-1, keepdim=True) / (std ** 4)
        
        stats_vector = torch.cat([mean, std, skew, kurt], dim=-1)
        params = self.stats_predictor(stats_vector)
        
        gamma = params[:, :, 0:1] 
        beta  = params[:, :, 1:2] 
        
        normalized = centered / std
        reprojected = normalized * gamma + beta
        
        return reprojected[:, :, self.window_size // 2]
