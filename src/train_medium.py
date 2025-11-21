import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import random

# Import your custom scripts
from data_loader import SaltDataset
from model import UNet

# ================= MEDIUM CONFIGURATION =================
# Tuned for ~45-60 minutes on CPU
BATCH_SIZE = 8          
LEARNING_RATE = 0.0005  
EPOCHS = 7              # Enough loops to learn basic shapes
SAMPLE_LIMIT = 800      # 800 random patches (instead of 5000+)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'CroppedData.segy')
LABEL_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'SaltMask.segy')
MASK_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'survey_mask.npy')
MODEL_SAVE_DIR = os.path.join(BASE_DIR, 'models')
# ========================================================

def train_medium():
    # Force CPU for consistency (unless you fixed CUDA)
    device = torch.device('cpu') 
    
    print(f"{'='*40}")
    print(f"⏳ STARTING MEDIUM TRAINING (~45-60 Mins) ⏳")
    print(f"   - Samples: {SAMPLE_LIMIT}")
    print(f"   - Epochs:  {EPOCHS}")
    print(f"{'='*40}")

    # 1. Load Dataset
    print("Loading Data...")
    full_dataset = SaltDataset(RAW_PATH, LABEL_PATH, MASK_PATH)
    
    # --- SUBSET STRATEGY ---
    # Pick 800 random indices so we get a variety of salt and no-salt
    # We sort them just to be tidy, though it doesn't matter for training
    all_indices = list(range(len(full_dataset)))
    random_indices = random.sample(all_indices, SAMPLE_LIMIT)
    
    medium_dataset = Subset(full_dataset, random_indices)
    
    loader = DataLoader(medium_dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Subset created: {len(medium_dataset)} samples selected randomly.")

    # 2. Setup Model
    model = UNet(n_channels=1, n_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    # 3. Training Loop
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        model.train()
        
        loop = tqdm(loader, desc=f"Epoch {epoch+1}")
        epoch_loss = 0
        
        for images, labels in loop:
            images = images.to(device)
            labels = labels.to(device)

            # Forward
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        # Print average loss for the epoch
        avg_loss = epoch_loss / len(loader)
        print(f"   > Avg Loss: {avg_loss:.4f}")

    # 4. Save Results
    save_path = os.path.join(MODEL_SAVE_DIR, 'medium_test_model.pth')
    torch.save(model.state_dict(), save_path)
    
    print(f"\n{'='*40}")
    print(f"✅ MEDIUM TRAINING COMPLETE")
    print(f"Model saved to: {save_path}")
    print(f"Now run 'src/inference.py' (change config to 'medium_test_model.pth')")

if __name__ == "__main__":
    train_medium()