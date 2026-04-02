import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from model import NeuralStatisticalGate

def generate_data(batch_size=32, seq_len=1000):
    """
    Generates synthetic data.
    - Signal: Sparse sine wave bursts (Structured, High Kurtosis).
    - Noise: Loud Gaussian static (Random, Low Kurtosis).
    """
    clean_batch = []
    noisy_batch = []
    
    for _ in range(batch_size):
        # 1. Clean Signal (Sparse Bursts)
        clean = np.zeros(seq_len)
        num_bursts = np.random.randint(2, 5)
        for _ in range(num_bursts):
            idx = np.random.randint(50, seq_len-100)
            width = np.random.randint(30, 80)
            t = np.linspace(0, np.pi * 6, width)
            envelope = np.sin(np.linspace(0, np.pi, width))
            clean[idx:idx+width] = np.sin(t) * envelope
            
        # 2. Noise (Non-Stationary)
        # Noise is sometimes louder (0.8) than signal (0.5) to test robustness
        noise_level = np.random.uniform(0.3, 0.8) 
        noise = np.random.normal(0, noise_level, seq_len)
        
        clean_batch.append(clean)
        noisy_batch.append(clean + noise)
        
    return torch.tensor(np.array(noisy_batch), dtype=torch.float32), \
           torch.tensor(np.array(clean_batch), dtype=torch.float32)

# --- Training Setup ---
model = NeuralStatisticalGate(window_size=50)
optimizer = optim.Adam(model.parameters(), lr=0.005)
loss_fn = nn.MSELoss()

print("Training Statistical Gate...")

# Training Loop
for i in range(800):
    noisy, clean = generate_data()
    
    # Forward pass
    pred = model(noisy)
    
    # Align targets (output is shorter due to windowing)
    # Output length = Input - Window + 1
    diff = clean.shape[1] - pred.shape[1]
    start = diff // 2
    target = clean[:, start : start + pred.shape[1]]
    
    # Loss & Update
    loss = loss_fn(pred, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if i % 100 == 0:
        print(f"Step {i}: Loss {loss.item():.5f}")

# --- Visualization ---
print("Generating demo plot...")
test_noisy, test_clean = generate_data(1, 1000)

with torch.no_grad():
    test_pred = model(test_noisy)

# Plotting
plt.figure(figsize=(10, 5))
valid_len = test_pred.shape[1]
start = (test_noisy.shape[1] - valid_len) // 2

# Plot Input (Gray)
plt.plot(test_noisy[0].numpy(), color='lightgray', label='Noisy Input')
# Plot Target (Black Dashed)
plt.plot(test_clean[0].numpy(), color='black', linestyle='--', alpha=0.5, label='Clean Target')
# Plot Output (Green)
plt.plot(np.arange(start, start+valid_len), test_pred[0].numpy(), color='green', linewidth=2, label='Denoised Output')

plt.legend()
plt.title("Statistical Gating: Rejecting High-Variance Noise")
plt.savefig("result.png")
plt.show()
print("Saved result.png")
