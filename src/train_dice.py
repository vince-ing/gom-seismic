import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split, ConcatDataset
from tqdm import tqdm
from model import UNet

# ================= CONFIG =================
CACHE_DIR = r"C:\Users\ig-gbds\Documents\SeismicProject\data\cache_chunks"
MODEL_SAVE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')

BATCH_SIZE = 12
LEARNING_RATE = 0.0001 # Slightly increased, we will use a scheduler
EPOCHS = 15 
# ==========================================

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # inputs: (B, C, H, W) softmax output
        # targets: (B, H, W) indices
        
        # We only care about the SALT class (Index 2) for Dice calculation
        # But for stability, we usually calculate it for all classes or specific ones.
        
        # One-hot encode targets to match input shape
        targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[1]).permute(0, 3, 1, 2).float()
        
        # Apply Softmax to inputs to get probabilities
        inputs_soft = F.softmax(inputs, dim=1)
        
        # Flatten
        inputs_flat = inputs_soft.contiguous().view(-1)
        targets_flat = targets_one_hot.contiguous().view(-1)
        
        intersection = (inputs_flat * targets_flat).sum()
        
        dice = (2.*intersection + smooth)/(inputs_flat.sum() + targets_flat.sum() + smooth)
        
        return 1 - dice

class CombinedLoss(nn.Module):
    def __init__(self, weight_ce=None):
        super(CombinedLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight_ce)
        self.dice = DiceLoss()

    def forward(self, inputs, targets):
        # inputs: Logits (Raw output from model)
        # targets: Class Indices
        
        loss_ce = self.ce(inputs, targets)
        loss_dice = self.dice(inputs, targets)
        
        # Weighted sum: We want shape accuracy (Dice) but need CE for stability
        # 0.5 CE + 0.5 Dice is standard, but here we prioritize Dice slightly
        return 0.4 * loss_ce + 0.6 * loss_dice

def train_cached():
    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"{'='*40}")
    print(f"🚀 STARTING BALANCED TRAINING (Dice + CE)")
    print(f"   Strategy: Optimize for Shape Overlap")
    print(f"{'='*40}")

    # 1. Load Data
    chunk_files = glob.glob(os.path.join(CACHE_DIR, "*.pt"))
    if not chunk_files:
        print("❌ No cache files found.")
        return

    datasets = []
    print(f"Loading {len(chunk_files)} chunks...")
    for f in tqdm(chunk_files):
        data = torch.load(f, weights_only=True)
        ds = TensorDataset(data['images'], data['labels'])
        datasets.append(ds)
    
    full_dataset = ConcatDataset(datasets)
    val_size = int(0.1 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 2. Model
    model = UNet(n_channels=1, n_classes=3).to(device)
    
    # BALANCED WEIGHTS:
    # We want Salt to be important, but not 500x more important than rock.
    # 0 (Rock): 1.0 (Baseline)
    # 1 (Water): 2.0 
    # 2 (Salt): 5.0 (Strong, but allows model to say "No" if it's clearly rock)
    weights = torch.tensor([1.0, 2.0, 5.0]).to(device)
    
    criterion = CombinedLoss(weight_ce=weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Scheduler: If loss plateaus, drop learning rate
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    best_val_loss = float('inf')

    # 3. Loop
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        model.train()
        loop = tqdm(train_loader, desc="Training")
        train_loss = 0
        
        salt_pred_count = 0
        
        for imgs, lbls in loop:
            imgs = imgs.to(device)
            lbls = lbls.to(device)

            # --- OPTIONAL: ON-THE-FLY NORMALIZATION ---
            # If your cached data isn't normalized, uncomment this:
            # mean = imgs.mean()
            # std = imgs.std()
            # imgs = (imgs - mean) / (std + 1e-8)
            # ------------------------------------------

            outputs = model(imgs)
            loss = criterion(outputs, lbls)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            
            # Monitoring: Count how many pixels are predicted as Salt (Class 2)
            preds = torch.argmax(outputs, dim=1)
            salt_pred_count += (preds == 2).sum().item()
            
            loop.set_postfix(loss=loss.item())

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs = imgs.to(device)
                lbls = lbls.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, lbls)
                val_loss += loss.item()
        
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        
        scheduler.step(avg_val)
        
        print(f"  > Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")
        print(f"  > Total Salt Pixels Predicted: {salt_pred_count} (Lower is better if previously saturated)")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, 'best_model.pth'))
            print("  > ⭐ Saved New Best Model")

    print("\n✅ Done!")

if __name__ == "__main__":
    train_cached()
