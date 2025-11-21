import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import random
import numpy as np

# Import your custom scripts
from data_loader import SaltDataset
from model import UNet

# ================= MEDIUM CONFIGURATION =================
BATCH_SIZE = 8          
LEARNING_RATE = 0.0005  
EPOCHS = 7              
SAMPLE_LIMIT = 800      
NUM_CLASSES = 3         # 0=Background, 1=Salt, 2=Other

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'CroppedData.segy')
LABEL_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'SaltMask.segy')
MASK_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'survey_mask.npy')
MODEL_SAVE_DIR = os.path.join(BASE_DIR, 'models')
# ========================================================

def compute_class_weights(dataset, device):
    """
    Scans the dataset subset to count pixels per class.
    Returns a weight tensor to balance the loss.
    """
    print("\n⚖️  Calculating Class Weights (scanning dataset)...")
    
    # Initialize counters for classes 0, 1, 2
    class_counts = torch.zeros(NUM_CLASSES)
    
    # We use a small loader to speed up the scan
    temp_loader = DataLoader(dataset, batch_size=8, shuffle=False)
    
    for _, labels in tqdm(temp_loader, desc="Scanning pixels"):
        # Flatten labels to count occurrences
        flattened = labels.view(-1).long()
        counts = torch.bincount(flattened, minlength=NUM_CLASSES)
        class_counts += counts

    total_pixels = class_counts.sum()
    
    # AVOID DIVISION BY ZERO: Add 1 to any count that is 0
    class_counts = torch.max(class_counts, torch.tensor(1.0))

    # FORMULA: Total / (Num_Classes * Count)
    # This is the standard "balanced" weight formula from sklearn
    weights = total_pixels / (NUM_CLASSES * class_counts)
    
    print(f"   > Pixel Counts: {class_counts.tolist()}")
    print(f"   > Calculated Weights: {weights.tolist()}")
    print(f"   (This means the model is penalized {weights[1]:.2f}x more for missing Salt)")
    
    return weights.to(device)

def train_medium():
    device = torch.device('cpu') 
    
    print(f"{'='*40}")
    print(f"⏳ STARTING MEDIUM TRAINING (Weighted Loss) ⏳")
    print(f"   - Samples: {SAMPLE_LIMIT}")
    print(f"   - Epochs:  {EPOCHS}")
    print(f"{'='*40}")

    # 1. Load Dataset
    print("Loading Data...")
    full_dataset = SaltDataset(RAW_PATH, LABEL_PATH, MASK_PATH)
    
    all_indices = list(range(len(full_dataset)))
    random_indices = random.sample(all_indices, SAMPLE_LIMIT)
    medium_dataset = Subset(full_dataset, random_indices)
    
    # 2. Calculate Weights BEFORE creating the main loader
    # This ensures we penalize the model correctly based on THIS specific subset
    class_weights = compute_class_weights(medium_dataset, device)

    loader = DataLoader(medium_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 3. Setup Model with Weighted Loss
    model = UNet(n_channels=1, n_classes=NUM_CLASSES).to(device)
    
    # PASS WEIGHTS HERE to fix the imbalance
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    # 4. Training Loop
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        model.train()
        
        loop = tqdm(loader, desc=f"Epoch {epoch+1}")
        epoch_loss = 0
        
        total_salt_pixels_predicted = 0
        
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
            
            # --- MONITORING ---
            # Check if the model is predicting ANY salt (Class 1)
            # dim=1 is the channel dimension for classes
            predictions = torch.argmax(outputs, dim=1)
            salt_pred_count = (predictions == 1).sum().item()
            total_salt_pixels_predicted += salt_pred_count
            
            loop.set_postfix(loss=loss.item(), salt_preds=salt_pred_count)

        avg_loss = epoch_loss / len(loader)
        print(f"   > Avg Loss: {avg_loss:.4f}")
        print(f"   > Total Salt Pixels Predicted this Epoch: {total_salt_pixels_predicted}")
        
        if total_salt_pixels_predicted == 0:
             print("   ⚠️ WARNING: Model is still predicting only background.")

    # 5. Save Results
    save_path = os.path.join(MODEL_SAVE_DIR, 'medium_weighted_model.pth')
    torch.save(model.state_dict(), save_path)
    
    print(f"\n{'='*40}")
    print(f"✅ MEDIUM TRAINING COMPLETE")
    print(f"Model saved to: {save_path}")

if __name__ == "__main__":
    train_medium()