import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from data_loader import SaltDataset
from model import UNet

# ================= CONFIGURATION =================
# Load the medium model you just trained
MODEL_NAME = 'medium_test_model.pth' 

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'CroppedData.segy')
LABEL_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'SaltMask.segy')
MASK_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'survey_mask.npy')
MODEL_PATH = os.path.join(BASE_DIR, 'models', MODEL_NAME)
OUTPUT_IMG = os.path.join(BASE_DIR, 'data', 'processed', 'prediction_comparison_medium.png')
# =================================================

def run_inference():
    print(f"{'='*40}")
    print(f"Loading Model: {MODEL_NAME}")
    print(f"{'='*40}")

    # 1. Setup Device
    device = torch.device('cpu')
    
    # 2. Load Model
    model = UNet(n_channels=1, n_classes=3).to(device)
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model file not found at {MODEL_PATH}")
        return
        
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("Model loaded.")

    # 3. Load Data
    ds = SaltDataset(RAW_PATH, LABEL_PATH, MASK_PATH)
    
    # 4. Setup Plot: 3 Rows (Samples) x 3 Columns (Views)
    fig, axs = plt.subplots(3, 3, figsize=(15, 12))
    plt.suptitle(f"Prediction Analysis: {MODEL_NAME}\n(Showing 3 Random Slices containing Salt)", fontsize=16)

    # Headers for columns
    cols = ['Raw Seismic', 'Ground Truth (Human)', 'AI Prediction (Model)']
    for ax, col in zip(axs[0], cols):
        ax.set_title(col, fontsize=14, fontweight='bold')

    print("Hunting for samples containing salt...")
    
    for row in range(3):
        # Hunt for a slice that actually has Salt (Class 2)
        # We try up to 50 times to find a good one, otherwise just take whatever
        for attempt in range(50):
            idx = torch.randint(0, len(ds), (1,)).item()
            image, mask = ds[idx]
            if np.any(mask.numpy() == 2): # Found salt!
                break
        
        # Predict
        input_tensor = image.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            prediction = torch.argmax(output, dim=1).squeeze().cpu().numpy()

        # --- COLUMN 1: SEISMIC ---
        axs[row, 0].imshow(image.squeeze(), cmap='gray')
        axs[row, 0].axis('off')
        axs[row, 0].set_ylabel(f"Sample {idx}", fontsize=12)

        # --- COLUMN 2: GROUND TRUTH ---
        # 0=Purple, 1=Teal, 2=Yellow
        axs[row, 1].imshow(mask, cmap='viridis', vmin=0, vmax=2)
        axs[row, 1].axis('off')

        # --- COLUMN 3: AI PREDICTION ---
        axs[row, 2].imshow(prediction, cmap='viridis', vmin=0, vmax=2)
        axs[row, 2].axis('off')

    plt.tight_layout()
    plt.savefig(OUTPUT_IMG)
    print(f"✅ Comparison saved to: {OUTPUT_IMG}")
    print("Open the image to compare the Middle Column (Truth) vs Right Column (AI).")

if __name__ == "__main__":
    run_inference()