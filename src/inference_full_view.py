import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from skimage import morphology
from skimage.morphology import disk
from matplotlib import colors
from data_loader import SaltDataset
from model import UNet

# ================= CONFIGURATION =================
MODEL_NAME = 'best_model.pth' 
NUM_SAMPLES = 5  # We only need a few full-height examples
PATCH_SIZE = (1024, 256) # Full vertical trace

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'CroppedData.segy')
LABEL_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'SaltMask.segy')
MASK_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'survey_mask.npy')
MODEL_PATH = os.path.join(BASE_DIR, 'models', MODEL_NAME)
OUTPUT_IMG = os.path.join(BASE_DIR, 'data', 'processed', 'prediction_full_water_salt.png')
# =================================================

def robust_normalize_batch(imgs):
    """ Matches training normalization EXACTLY. """
    mean = imgs.mean(dim=(2, 3), keepdim=True)
    std = imgs.std(dim=(2, 3), keepdim=True)
    
    # Clip outliers
    lower_limit = mean - 2.5 * std
    upper_limit = mean + 2.5 * std
    imgs = torch.clamp(imgs, min=lower_limit, max=upper_limit)
    
    # Scale to [-1, 1]
    min_val = imgs.amin(dim=(2, 3), keepdim=True)
    max_val = imgs.amax(dim=(2, 3), keepdim=True)
    imgs = 2 * (imgs - min_val) / (max_val - min_val + 1e-6) - 1.0
    return imgs

def run_inference_full():
    print(f"{'='*40}")
    print(f"🔍 FULL VIEW DIAGNOSTIC: Water + Salt")
    print(f"{'='*40}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Model
    model = UNet(n_channels=1, n_classes=3).to(device)
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model file not found at {MODEL_PATH}")
        return
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("✅ Model loaded.")

    # Load Dataset
    print("📂 Loading Dataset...")
    ds = SaltDataset(RAW_PATH, LABEL_PATH, MASK_PATH, patch_size=PATCH_SIZE)
    
    # Setup Plot: TALL figure for 1024px height
    # 5 Rows, 3 Columns. Height is exaggerated to show detail.
    fig, axs = plt.subplots(NUM_SAMPLES, 3, figsize=(15, 6 * NUM_SAMPLES))
    plt.suptitle(f"Deep Learning Analysis: Water Column vs. Salt Body\n(Model: {MODEL_NAME})", fontsize=16)

    cols = ['Raw Seismic', 'Ground Truth\n(Purple=Rock, Green=Water, Yellow=Salt)', 'Predicted Salt (Cleaned)']
    for ax, col in zip(axs[0], cols):
        ax.set_title(col, fontsize=14, fontweight='bold')

    print(f"Hunting for {NUM_SAMPLES} samples containing BOTH Water (1) and Salt (2)...")
    
    samples_found = 0
    attempts = 0
    
    # Random shuffle indices to find diverse examples
    indices = np.arange(len(ds))
    np.random.shuffle(indices)

    for idx in indices:
        if samples_found >= NUM_SAMPLES:
            break
            
        attempts += 1
        
        # Load raw data
        image, mask = ds[idx] # image is (1, 1024, 256), mask is (1024, 256)
        
        # === FILTER: MUST HAVE WATER AND SALT ===
        unique_classes = np.unique(mask.numpy())
        has_water = 1 in unique_classes
        has_salt = 2 in unique_classes
        
        if has_water and has_salt:
            
            # Prepare Input
            input_tensor = image.unsqueeze(0).to(device)
            input_tensor = robust_normalize_batch(input_tensor)
            
            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.softmax(output, dim=1)
                salt_prob = probs[0, 2, :, :].cpu().numpy()

            # === CLEANUP STAGE ===
            binary_mask = salt_prob > 0.5
            clean_mask = morphology.remove_small_objects(binary_mask, min_size=64)
            clean_mask = morphology.binary_closing(clean_mask, disk(3))
            final_prediction = clean_mask.astype(float)

            # === PLOTTING ===
            row = samples_found
            
            # 1. Seismic
            axs[row, 0].imshow(image.squeeze(), cmap='gray', aspect='auto')
            axs[row, 0].set_ylabel(f"Sample {idx}\n(Depth 0-1024)", fontsize=12)

            # 2. Ground Truth (Custom Colormap)
            # 0=Purple, 1=Green, 2=Yellow
            cmap_gt = colors.ListedColormap(['#440154', '#21918c', '#fde725']) # Viridis-like discrete
            axs[row, 1].imshow(mask, cmap=cmap_gt, vmin=0, vmax=2, aspect='auto')

            # 3. Prediction
            axs[row, 2].imshow(final_prediction, cmap='plasma', vmin=0, vmax=1, aspect='auto')
            
            samples_found += 1
            print(f"✅ Found Sample {idx}: Has Water & Salt ({samples_found}/{NUM_SAMPLES})")
        
        if attempts % 500 == 0:
            print(f"   ...checked {attempts} images...")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(OUTPUT_IMG)
    print(f"\n📸 Image saved to: {OUTPUT_IMG}")

if __name__ == "__main__":
    run_inference_full()
