import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict
from model import NeuralStatisticalGate

class ChannelEnvironmentSimulator:
    """
    Simulates a specific channel environment. 
    The trained model will be optimized specifically for this environment's noise profile.
    """
    @staticmethod
    def generate_environment_data(batch_size: int = 32, seq_len: int = 1000) -> Tuple[torch.Tensor, torch.Tensor]:
        clean_batch, noisy_batch = [], []
        
        for _ in range(batch_size):
            clean = np.zeros(seq_len)
            num_bursts = np.random.randint(2, 5)
            
            for _ in range(num_bursts):
                idx = np.random.randint(50, seq_len - 100)
                width = np.random.randint(30, 80)
                t = np.linspace(0, np.pi * 6, width)
                envelope = np.sin(np.linspace(0, np.pi, width))
                clean[idx:idx+width] = np.sin(t) * envelope
                
            # SPECIFIC ENVIRONMENT Caution: Signal against which it is trained must be in the same 
            # environment and signals for filtering also must be in the same environment.
            noise_level = np.random.uniform(0.3, 0.8) 
            noise = np.random.normal(0, noise_level, seq_len)
            
            clean_batch.append(clean)
            noisy_batch.append(clean + noise)
            
        return torch.tensor(np.array(noisy_batch), dtype=torch.float32), \
               torch.tensor(np.array(clean_batch), dtype=torch.float32)

def evaluate_performance(clean: torch.Tensor, noisy: torch.Tensor, denoised: torch.Tensor) -> Dict[str, float]:
    """Calculates environment-specific SNR improvement."""
    valid_len = denoised.shape[1]
    start = (clean.shape[1] - valid_len) // 2
    
    clean_aligned = clean[:, start : start + valid_len]
    noisy_aligned = noisy[:, start : start + valid_len]
    
    rmse = torch.sqrt(torch.mean((clean_aligned - denoised)**2)).item()
    power_clean = torch.mean(clean_aligned**2)
    power_input_noise = torch.mean((noisy_aligned - clean_aligned)**2)
    snr_in = 10 * torch.log10(power_clean / (power_input_noise + 1e-8))
    
    power_output_noise = torch.mean((denoised - clean_aligned)**2)
    snr_out = 10 * torch.log10(power_clean / (power_output_noise + 1e-8))
    
    return {
        "RMSE": rmse,
        "SNR_Improvement_dB": (snr_out - snr_in).item()
    }


def run_environment_simulation():
    """Trains and validates the model for the specific channel environment."""
    model = NeuralStatisticalGate(window_size=50)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    loss_fn = nn.MSELoss()

    print("Initializing Specific Channel Environment Simulation...\n")
    
    for step in range(801):
        noisy, clean = ChannelEnvironmentSimulator.generate_environment_data()
        pred = model(noisy)
        
        diff = clean.shape[1] - pred.shape[1]
        start = diff // 2
        target = clean[:, start : start + pred.shape[1]]
        
        loss = loss_fn(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 100 == 0:
            with torch.no_grad():
                val_noisy, val_clean = ChannelEnvironmentSimulator.generate_environment_data()
                val_pred = model(val_noisy)
                metrics = evaluate_performance(val_clean, val_noisy, val_pred)
                
            print(f"Step {step:03d} | Train MSE: {loss.item():.4f} | "
                  f"Validation SNR Imprv: +{metrics['SNR_Improvement_dB']:.2f} dB")
            
    return model

if __name__ == "__main__":
    trained_model = run_environment_simulation()
    print("\nSimulation complete. Note: Model weights are calibrated strictly for the simulated environment.")
