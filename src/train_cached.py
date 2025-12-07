import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split, ConcatDataset
from tqdm import tqdm
from model import UNet

# ================= CONFIGURATION =================
CACHE_DIR = r"C:\Users\ig-gbds\Documents\SeismicProject\data\cache_chunks"
MODEL_SAVE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')

BATCH_SIZE = 12
LEARNING_RATE = 1e-4 
EPOCHS = 20          
# =================================================

class FocalTverskyLoss(nn.Module):
    """
    Implements Focal Tversky Loss as defined in Section 6.3 of the report[cite: 208].
    Optimizes for Recall on the minority class to prevent null-prediction mode collapse.
    """
    def __init__(self, alpha=0.3, beta=0.7, gamma=1.33, smooth=1e-6):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha  # Penalizes False Positives [cite: 211]
        self.beta = beta    # Penalizes False Negatives (Recall focus) [cite: 212]
        self.gamma = gamma  # Focusing parameter for hard examples [cite: 127]
        self.smooth = smooth

    def forward(self, inputs, targets):
        # inputs: (B, C, H, W) logits
        # targets: (B, H, W) class indices
        
        # Softmax to get probabilities
        inputs_soft = F.softmax(inputs, dim=1)
        
        # Isolate Salt Class (Index 2)
        salt_inputs = inputs_soft[:, 2, :, :]       
        salt_targets = (targets == 2).float() 

        # Flatten spatial dimensions only (B, -1) to preserve batch statistics [cite: 221]
        inputs_flat = salt_inputs.contiguous().view(salt_inputs.size(0), -1)
        targets_flat = salt_targets.contiguous().view(salt_targets.size(0), -1)
        
        # Tversky Components
        TP = (inputs_flat * targets_flat).sum(1)
        FP = ((1-targets_flat) * inputs_flat).sum(1)
        FN = (targets_flat * (1-inputs_flat)).sum(1)
        
        # Tversky Index [cite: 114]
        tversky = (TP + self.smooth) / (TP + self.alpha*FP + self.beta*FN + self.smooth)
        
        # Focal Modulation [cite: 126]
        loss = (1 - tversky)**self.gamma
        
        return loss.mean()

def robust_normalize_batch(imgs):
    """
    Applies Per-Image Percentile Clipping (2nd/98th) as defined in Section 6.2[cite: 180].
    Mitigates gradient explosion caused by high-amplitude water bottom artifacts.
    """
    # 1. Estimate percentiles using mean/std approximation for computational efficiency [cite: 194]
    # We keepdim=True to calculate stats per-image, not globally [cite: 194]
    mean = imgs.mean(dim=(2, 3), keepdim=True)
    std = imgs.std(dim=(2, 3), keepdim=True)
    
    # 2. Clip at approx 2.5 std devs (covering ~98% of data) [cite: 195]
    lower_limit = mean - 2.5 * std
    upper_limit = mean + 2.5 * std
    imgs = torch.clamp(imgs, min=lower_limit, max=upper_limit)
    
    # 3. Min-Max Scale the clipped data to [-1, 1] [cite: 203]
    min_val = imgs.amin(dim=(2, 3), keepdim=True)
    max_val = imgs.amax(dim=(2, 3), keepdim=True)
    
    # Avoid division by zero with epsilon [cite: 203]
    imgs = 2 * (imgs - min_val) / (max_val - min_val + 1e-6) - 1.0
    return imgs

def train_cached():
    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Initializing Focal Tversky Training Strategy[cite: 11]...")

    # 1. Load Data
    chunk_files = glob.glob(os.path.join(CACHE_DIR, "*.pt"))
    if not chunk_files:
        print("Error: No cache files found.")
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

    # 2. Model & Optimization
    model = UNet(n_channels=1, n_classes=3).to(device)
    
    # Focal Tversky Loss with beta=0.7 to prioritize Recall [cite: 123]
    criterion = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=1.33)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Cosine Annealing scheduler to escape local minima [cite: 254]
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5)
    
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    best_val_loss = float('inf')

    # 3. Training Loop
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        model.train()
        loop = tqdm(train_loader, desc="Training")
        train_loss = 0
        salt_pixels_found = 0
        
        for imgs, lbls in loop:
            imgs = imgs.to(device)
            lbls = lbls.to(device)

            # Apply Per-Image Robust Normalization [cite: 237]
            imgs = robust_normalize_batch(imgs)

            outputs = model(imgs)
            loss = criterion(outputs, lbls)

            optimizer.zero_grad()
            loss.backward()
            
            # Gradient Clipping to prevent instability [cite: 245]
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
            
            # Monitor Salt Predictions (Class 2)
            preds = torch.argmax(outputs, dim=1)
            salt_pixels_found += (preds == 2).sum().item()
            
            loop.set_postfix(loss=loss.item())

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs = imgs.to(device)
                lbls = lbls.to(device)
                imgs = robust_normalize_batch(imgs)
                
                outputs = model(imgs)
                loss = criterion(outputs, lbls)
                val_loss += loss.item()
        
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        
        scheduler.step(epoch + avg_val) # Step for Cosine Annealing
        
        print(f"  > Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")
        print(f"  > Salt Pixels Detected: {salt_pixels_found}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, 'best_model.pth'))
            print("  > Saved Best Model")

    print("Training Complete.")

if __name__ == "__main__":
    train_cached()
