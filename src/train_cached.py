import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split, ConcatDataset
from tqdm import tqdm
from model import UNet

# ================= FOCAL CONFIG =================
CACHE_DIR = r"C:\Users\ig-gbds\Documents\SeismicProject\data\cache_chunks"
MODEL_SAVE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')

BATCH_SIZE = 12
LEARNING_RATE = 0.00005  # Lower LR for stability
EPOCHS = 6
# ================================================

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: (B, C, H, W) -> Softmax probabilities
        # targets: (B, H, W)
        
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss) # Probability of the correct class
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        else:
            return focal_loss.sum()

def train_cached():
    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"{'='*40}")
    print(f"🚀 STARTING FOCAL LOSS TRAINING")
    print(f"   Strategy: Force model to learn Salt")
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
    
    # NUCLEAR WEIGHTS:
    # 0 (Rock): 0.1 (Don't care)
    # 1 (Water): 2.0 (Standard)
    # 2 (Salt): 50.0 (CRITICAL PRIORITY)
    weights = torch.tensor([0.1, 2.0, 50.0]).to(device)
    
    # Use Focal Loss instead of Cross Entropy
    criterion = FocalLoss(alpha=weights, gamma=2.0)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    best_val_loss = float('inf')

    # 3. Loop
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        model.train()
        loop = tqdm(train_loader, desc="Training")
        train_loss = 0
        
        # Monitor Salt
        salt_pixels_found = 0
        
        for imgs, lbls in loop:
            imgs = imgs.to(device)
            lbls = lbls.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, lbls)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            
            # Check if it predicts ANY salt
            preds = torch.argmax(outputs, dim=1)
            salt_pixels_found += (preds == 2).sum().item()
            
            loop.set_postfix(loss=loss.item(), salt=salt_pixels_found)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs = imgs.to(device)
                lbls = lbls.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, lbls)
                val_loss += loss.item()
        
        avg_val = val_loss / len(val_loader)
        print(f"  > Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, 'best_model.pth'))
            print("  > ⭐ Saved New Best Model")

    print("\n✅ Done!")

if __name__ == "__main__":
    train_cached()
