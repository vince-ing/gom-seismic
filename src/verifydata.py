import torch
import matplotlib.pyplot as plt
import os
from data_loader import SaltDataset

# ================= CONFIGURATION =================
# Define paths (Assumes you are running from G:\SeismicProject)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'CroppedData.segy')
LABEL_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'SaltMask.segy')
MASK_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'survey_mask.npy')
OUTPUT_IMG = os.path.join(BASE_DIR, 'data', 'processed', 'loader_verification.png')
# =================================================

def verify():
    print(f"{'='*40}")
    print("STEP 1: Verifying Data Loader")
    print(f"{'='*40}")

    # 1. Check files
    if not os.path.exists(MASK_PATH):
        print("❌ ERROR: survey_mask.npy not found!")
        print("   -> Run 'python src/make_mask_v2.py' first.")
        return

    # 2. Initialize Dataset
    print("Initializing Dataset...")
    try:
        ds = SaltDataset(RAW_PATH, LABEL_PATH, MASK_PATH)
        print(f"✅ Success! Found {len(ds)} valid inlines.")
    except Exception as e:
        print(f"❌ Error initializing dataset: {e}")
        return

    # 3. Grab 4 random samples
    print("Grabbing 4 random patches...")
    fig, axs = plt.subplots(2, 4, figsize=(15, 8))
    
    for i in range(4):
        # Data Loader returns: (image_tensor, label_tensor)
        # Image: (1, 256, 256) -> we squeeze to (256, 256)
        # Label: (256, 256)
        try:
            idx = torch.randint(0, len(ds), (1,)).item()
            img, lbl = ds[idx]
            
            # Plot Seismic
            axs[0, i].imshow(img.squeeze(), cmap='gray')
            axs[0, i].set_title(f"Seismic (Patch {i})")
            axs[0, i].axis('off')
            
            # Plot Label
            # 0=Purple, 1=Teal, 2=Yellow
            axs[1, i].imshow(lbl, cmap='viridis', vmin=0, vmax=2)
            axs[1, i].set_title(f"Label (Patch {i})")
            axs[1, i].axis('off')
            
        except Exception as e:
            print(f"❌ Error grabbing patch {i}: {e}")

    plt.tight_layout()
    plt.savefig(OUTPUT_IMG)
    print(f"\n✅ Verification Image Saved: {OUTPUT_IMG}")
    print("Open that image. If you see seismic waves on top and colored blobs on bottom, YOU ARE READY.")

if __name__ == "__main__":
    verify()