import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# Import your custom scripts
from data_loader import SaltDataset
from model import UNet

# ================= FAST CONFIGURATION =================
BATCH_SIZE = 4          
LEARNING_RATE = 0.0001  
EPOCHS = 2              # Only do 2 loops
SAMPLE_LIMIT = 20       # Only look at 20 random patches (instead of 5000)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'CroppedData.segy')
LABEL_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'SaltMask.segy')
MASK_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'survey_mask.npy')
MODEL_SAVE_DIR = os.path.join(BASE_DIR, 'models')
# ======================================================

def train_fast():
    device = torch.device('cpu') # Force CPU for stability in test
    print(f"{'='*40}")
    print(f"⚡ STARTING FAST TEST (2 Epochs, 20 Samples) ⚡")
    print(f"{'='*40}")

    # 1. Load Dataset
    print("Loading Data...")
    full_dataset = SaltDataset(RAW_PATH, LABEL_PATH, MASK_PATH)
    
    # --- CHEAT: Create a tiny subset of data ---
    # We only take the first 20 valid patches
    indices = range(SAMPLE_LIMIT) 
    small_dataset = Subset(full_dataset, indices)
    
    loader = DataLoader(small_dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Subset created: {len(small_dataset)} samples.")

    # 2. Setup Model
    model = UNet(n_channels=1, n_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    # 3. Tiny Training Loop
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        model.train()
        
        loop = tqdm(loader, desc="Fast Train")
        
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

            loop.set_postfix(loss=loss.item())

    # 4. Test Saving
    save_path = os.path.join(MODEL_SAVE_DIR, 'fast_test_model.pth')
    torch.save(model.state_dict(), save_path)
    print(f"\n{'='*40}")
    print(f"✅ SUCCESS! Pipeline is valid.")
    print(f"Test model saved to: {save_path}")
    print(f"You can now run the real 'src/train.py' overnight.")

if __name__ == "__main__":
    train_fast()