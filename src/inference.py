import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from data_loader import SaltDataset
from model import UNet
from skimage import morphology
from skimage.morphology import disk

# ================= CONFIGURATION =================
MODEL_NAME = 'best_model.pth' 
NUM_SAMPLES = 9 

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'CroppedData.segy')
LABEL_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'SaltMask.segy')
MASK_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'survey_mask.npy')
MODEL_PATH = os.path.join(BASE_DIR, 'models', MODEL_NAME)
OUTPUT_IMG = os.path.join(BASE_DIR, 'data', 'processed', 'prediction_debug.png')
# =================================================

def robust_normalize_batch(imgs):
    """
    MUST MATCH TRAINING EXACTLY.
    Clips outliers and scales to [-1, 1].
    """
    # imgs shape: (B, C, H, W)
    mean = imgs.mean(dim=(2, 3), keepdim=True)
    std = imgs.std(dim=(2, 3), keepdim=True)
    
    # Clip at approx 2.5 std devs (covering ~98% of data)
    lower_limit = mean - 2.5 * std
    upper_limit = mean + 2.5 * std
    imgs = torch.clamp(imgs, min=lower_limit, max=upper_limit)
    
    # Min-Max Scale the clipped data to [-1, 1]
    min_val = imgs.amin(dim=(2, 3), keepdim=True)
    max_val = imgs.amax(dim=(2, 3), keepdim=True)
    
    # Avoid division by zero
    imgs = 2 * (imgs - min_val) / (max_val - min_val + 1e-6) - 1.0
    return imgs

def run_inference():
    print(f"{'='*40}")
    print(f"Loading Model: {MODEL_NAME}")
    print(f"{'='*40}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(n_channels=1, n_classes=3).to(device)
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model file not found at {MODEL_PATH}")
        return
        
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("Model loaded.")

    # Load Data (Use 1024x256 to match training)
    ds = SaltDataset(RAW_PATH, LABEL_PATH, MASK_PATH, patch_size=(1024, 256))
    
    # Setup Plot: 9 Rows x 3 Columns
    fig, axs = plt.subplots(NUM_SAMPLES, 3, figsize=(12, 4 * NUM_SAMPLES))
    plt.suptitle(f"DEBUG Analysis: {MODEL_NAME}\n(Showing Salt Probability Map)", fontsize=16)

    cols = ['Raw Seismic', 'True Salt Mask', 'Predicted Salt Probability']
    for ax, col in zip(axs[0], cols):
        ax.set_title(col, fontsize=14, fontweight='bold')

    print(f"Hunting for {NUM_SAMPLES} samples containing salt...")
    
    samples_found = 0
    attempts = 0
    
    while samples_found < NUM_SAMPLES:
        attempts += 1
        idx = torch.randint(0, len(ds), (1,)).item()
        
        # ds[idx] returns (C, H, W)
        image, mask = ds[idx]
        
        # Check if sample has salt (Class 2)
        if np.any(mask.numpy() == 2):
            
            # Prepare Input
            # 1. Add Batch Dimension: (1, 1, 1024, 256)
            input_tensor = image.unsqueeze(0).to(device)
            
            # 2. APPLY THE FIX: Normalize exactly like training
            input_tensor = robust_normalize_batch(input_tensor)
            
            with torch.no_grad():
                output = model(input_tensor) # (1, 3, H, W)
                probs = torch.softmax(output, dim=1)
                
                # Extract just the SALT channel (Index 2)
                salt_prob = probs[0, 2, :, :].cpu().numpy()

            #Clean up
            # 1. Threshold: Make it solid Salt (1) or Rock (0)
            binary_mask = salt_prob > 0.5

            # 2. Despeckle: Remove small yellow noise (fewer than 64 pixels)
            clean_mask = morphology.remove_small_objects(binary_mask, min_size=64)

            # 3. Fill Holes: Close small black gaps inside the salt
            clean_mask = morphology.binary_closing(clean_mask, disk(3))

            # 4. Update the variable so the plot shows the CLEAN version
            salt_prob = clean_mask.astype(float)

            # --- PLOTTING ---
            row = samples_found
            
            # Col 1: Seismic (Squeeze to 2D for plotting)
            axs[row, 0].imshow(image.squeeze(), cmap='gray', aspect='auto')
            axs[row, 0].set_ylabel(f"Sample {idx}", fontsize=12)

            # Col 2: Ground Truth (Show only Salt as Yellow)
            salt_mask = (mask.numpy() == 2).astype(float)
            axs[row, 1].imshow(salt_mask, cmap='viridis', aspect='auto')

            # Col 3: Salt Probability Heatmap
            im = axs[row, 2].imshow(salt_prob, cmap='plasma', vmin=0, vmax=1, aspect='auto')
            
            if row == 0:
                plt.colorbar(im, ax=axs[row, 2], label="Salt Probability")
            
            samples_found += 1
            print(f"Found sample {samples_found}/{NUM_SAMPLES}")
            
        if attempts > 2000:
            print("⚠️ Warning: Could not find enough salt samples.")
            break

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(OUTPUT_IMG)
    print(f"✅ Debug image saved to: {OUTPUT_IMG}")

if __name__ == "__main__":
    run_inference()
