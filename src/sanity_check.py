import torch
import torch.nn as nn
import torch.optim as optim
from model import UNet

# 1. Fake Data that looks like yours (Huge amplitudes)
# Simulating seismic data with values between -2000 and +2000
raw_input = torch.randn(8, 1, 256, 256) * 2000 
# Fake labels (0, 1, 2)
labels = torch.randint(0, 3, (8, 256, 256)).long() 

def normalize_batch(imgs):
    # THE FIX: Force data into [-1, 1] range
    mean = imgs.mean(dim=(2, 3), keepdim=True)
    std = imgs.std(dim=(2, 3), keepdim=True)
    return (imgs - mean) / (std + 1e-8)

def sanity_check():
    print("🧪 Running 2-Minute Sanity Check...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(n_channels=1, n_classes=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Move to GPU
    data = raw_input.to(device)
    target = labels.to(device)
    
    print(f"   Input Stats Before Fix: Min={data.min():.2f}, Max={data.max():.2f}")
    
    # Apply Fix
    data_normalized = normalize_batch(data)
    print(f"   Input Stats After Fix:  Min={data_normalized.min():.2f}, Max={data_normalized.max():.2f}")
    print("\n   Starting Training Loop (Target: Loss -> 0)...")

    # Train on THIS ONE BATCH 50 times
    for i in range(50):
        optimizer.zero_grad()
        output = model(data_normalized)
        loss = criterion(output, target)
        loss.backward()
        
        # Apply Safety Clip
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        if i % 10 == 0:
            print(f"   Iter {i}: Loss {loss.item():.4f}")

    print(f"   Iter 50: Loss {loss.item():.4f}")
    
    if loss.item() < 0.1:
        print("\n✅ SUCCESS: The fix works. The model learned the batch.")
    else:
        print("\n❌ FAILURE: The model is still broken.")

if __name__ == "__main__":
    sanity_check()
