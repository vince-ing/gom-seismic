import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm  # Progress bar library

# Import your custom scripts
from data_loader import SaltDataset
from model import UNet

# ================= CONFIGURATION =================
# Hyperparameters
BATCH_SIZE = 8          # How many patches to process at once (Lower if "Out of Memory" error)
LEARNING_RATE = 0.0001  # How fast the model learns (Low is safer)
EPOCHS = 10             # How many times to look at the entire dataset
VALIDATION_SPLIT = 0.1  # 10% of data used for testing, 90% for training

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'CroppedData.segy')
LABEL_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'SaltMask.segy')
MASK_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'survey_mask.npy')
MODEL_SAVE_DIR = os.path.join(BASE_DIR, 'models')
# =================================================

def train():
    # 1. Setup Device (GPU if available, else CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"{'='*40}")
    print(f"Starting Training on: {device}")
    print(f"{'='*40}")

    # 2. Prepare Data
    print("Loading Dataset...")
    full_dataset = SaltDataset(RAW_PATH, LABEL_PATH, MASK_PATH)
    
    # Split into Train (90%) and Validation (10%)
    val_size = int(len(full_dataset) * VALIDATION_SPLIT)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"Training Samples:   {train_size}")
    print(f"Validation Samples: {val_size}")

    # 3. Initialize Model, Loss, Optimizer
    model = UNet(n_channels=1, n_classes=3).to(device)
    
    # Weights: Salt (Class 2) is rare, so we can weight it higher if needed.
    # For now, standard CrossEntropy is fine.
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Ensure model directory exists
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    # 4. Training Loop
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        # --- TRAINING PHASE ---
        model.train()
        train_loss = 0
        loop = tqdm(train_loader, desc="Training")
        
        for images, labels in loop:
            images = images.to(device)
            labels = labels.to(device)

            # Forward Pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward Pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"  > Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # --- SAVE CHECKPOINT ---
        # Only save if this is the best model so far
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(MODEL_SAVE_DIR, 'best_model.pth')
            torch.save(model.state_dict(), save_path)
            print(f"  > ⭐ New Best Model saved to: {save_path}")

    print(f"\n{'='*40}")
    print("Training Complete!")
    print(f"Best Model is at: {os.path.join(MODEL_SAVE_DIR, 'best_model.pth')}")

if __name__ == "__main__":
    train()