import torch
import torch.nn as nn

class RobustStatisticalReprojector(nn.Module):
    """
    A Differentiable Statistical Filter that uses Higher-Order Moments
    (Mean, Variance, Skewness, Kurtosis) to gate signal from noise.
    """
    def __init__(self, window_size=50, hidden_dim=32):
        super().__init__()
        self.window_size = window_size
        
        # The "Brain": Maps Statistics -> Filter Parameters
        self.stats_predictor = nn.Sequential(
            nn.Linear(4, hidden_dim * 2),  # Expand features
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)       # Output: Gamma (Scale), Beta (Shift)
        )

    def forward(self, x):
        """
        Input:  x [Batch, Length]
        Output: y [Batch, Length] (Denoised)
        """
        # 1. Unfold into overlapping windows
        # shape: [Batch, Num_Windows, Window_Size]
        windows = x.unfold(1, self.window_size, 1)
        
        # 2. Extract Differentiable Statistics
        mean = windows.mean(dim=-1, keepdim=True)
        std = windows.std(dim=-1, keepdim=True) + 1e-6
        
        # Center the window (remove DC/Mean)
        centered = windows - mean
        
        # Skewness (3rd Moment)
        skew = (centered ** 3).mean(dim=-1, keepdim=True) / (std ** 3)
        
        # Kurtosis (4th Moment)
        kurt = (centered ** 4).mean(dim=-1, keepdim=True) / (std ** 4)
        
        # 3. Predict Adaptive Parameters
        # Vector: [Mean, Std, Skew, Kurt]
        stats_vector = torch.cat([mean, std, skew, kurt], dim=-1)
        params = self.stats_predictor(stats_vector)
        
        gamma = params[:, :, 0:1] # Learned Scale
        beta  = params[:, :, 1:2] # Learned Shift
        
        # 4. Re-project (Normalize -> Scale -> Shift)
        normalized = centered / std
        reprojected = normalized * gamma + beta
        
        # 5. Overlap-Add / Reconstruction
        # A full version would use torch.nn.functional.fold
        return reprojected[:, :, self.window_size//2]
