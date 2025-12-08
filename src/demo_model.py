import os
import torch
import numpy as np
import segyio
import matplotlib.pyplot as plt
from skimage import morphology
from skimage.morphology import disk
from scipy.ndimage import zoom
from model import UNet

# ================= USER CONFIGURATION =================
INPUT_FOLDER = r"G:\Working\Students\Undergraduate\For_Vince\SeismicFacies\namss.B-26-99-TX.mcs3d.airgun\Data\segy"
OUTPUT_FOLDER = r"G:\Working\Students\Undergraduate\For_Vince\SeismicFacies\Reports"

OUTPUT_FILENAME = "Salt_Discovery_Report.png"
OUTPUT_IMAGE = os.path.join(OUTPUT_FOLDER, OUTPUT_FILENAME)

MODEL_PATH = "models/best_model.pth"

# PHYSICS CORRECTION
SCALE_FACTOR = 0.4  # Resample 4ms -> 10ms
# =================================================

def robust_normalize_tile(img):
    mean = np.mean(img)
    std = np.std(img)
    img = np.clip(img, mean - 2.5 * std, mean + 2.5 * std)
    min_val, max_val = np.min(img), np.max(img)
    if max_val - min_val == 0: return np.zeros_like(img)
    return 2 * (img - min_val) / (max_val - min_val) - 1.0

class SeismicHunter:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"   🚀 Loading Model...")
        self.model = UNet(n_channels=1, n_classes=3).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def predict_slice(self, slice_data):
        # 1. Resample Vertical Axis
        slice_resampled = zoom(slice_data, (SCALE_FACTOR, 1.0), order=1)
        
        # 2. Crop/Pad to 1024
        h_new = slice_resampled.shape[0]
        if h_new > 1024:
            slice_resampled = slice_resampled[:1024, :]
        elif h_new < 1024:
            pad = np.zeros((1024, slice_resampled.shape[1]))
            pad[:h_new, :] = slice_resampled
            slice_resampled = pad
            
        # 3. Pad Width
        h, w = slice_resampled.shape
        target_w = ((w + 31) // 32) * 32
        pad_w = target_w - w
        if pad_w > 0: slice_resampled = np.pad(slice_resampled, ((0,0), (0, pad_w)), mode='constant')
            
        # 4. Predict
        slice_norm = robust_normalize_tile(slice_resampled)
        tensor = torch.from_numpy(slice_norm).float().unsqueeze(0).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            out = self.model(tensor)
            prob = torch.softmax(out, dim=1)
            salt_prob = prob[0, 2, :, :].cpu().numpy()
            
        # 5. Cleanup
        binary = salt_prob > 0.5
        clean = morphology.remove_small_objects(binary, min_size=200)
        
        # 6. Restore Original Dimensions
        clean = clean[:, :w]
        slice_norm = slice_norm[:, :w]
        
        clean_orig_scale = zoom(clean, (1/SCALE_FACTOR, 1.0), order=0)
        
        # Match dimensions
        clean_final = np.zeros(slice_data.shape)
        h_c = min(slice_data.shape[0], clean_orig_scale.shape[0])
        clean_final[:h_c, :] = clean_orig_scale[:h_c, :]
        
        return clean_final, slice_data 

def main():
    if not os.path.exists(INPUT_FOLDER): return
    hunter = SeismicHunter(MODEL_PATH)
    files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith('.segy') or f.endswith('.sgy')]
    
    import random
    random.shuffle(files)
    
    top_hits = []
    
    print(f"   🔎 EMERGENCY MODE: Brute Force List Reading...")

    for f in files:
        if len(top_hits) >= 3: break
        filepath = os.path.join(INPUT_FOLDER, f)
        
        try:
            with segyio.open(filepath, "r", ignore_geometry=True) as src:
                # Disable mmap on network drives to prevent locking issues
                # src.mmap() 
                n_traces = src.tracecount
                
                for _ in range(50):
                    start_idx = np.random.randint(0, max(1, n_traces - 2000))
                    end_idx = start_idx + 1000
                    
                    # === FIX: EXPLICIT LIST COMPREHENSION ===
                    # This forces the data into memory instantly. No generators.
                    # It is slower but bulletproof.
                    chunk = [src.trace[i] for i in range(start_idx, end_idx)]
                    
                    if len(chunk) < 500: continue
                    
                    # Stack and Transpose -> (Depth, Traces)
                    slice_data = np.stack(chunk).T 
                    
                    # Predict
                    mask_pred, seismic = hunter.predict_slice(slice_data)
                    
                    # SCORE
                    salt_score = np.sum(mask_pred)
                    
                    if salt_score > 2000:
                        print(f"   ✅ FOUND SALT! File: {f} (Score: {int(salt_score)})")
                        
                        try:
                            il = src.header[start_idx][segyio.TraceField.INLINE_3D]
                        except:
                            il = "Unknown"

                        top_hits.append({
                            'file': f,
                            'inline': il,
                            'seismic': seismic,
                            'mask': mask_pred,
                            'score': salt_score
                        })
                        break 
                        
        except Exception as e:
            print(f"   ⚠️ Skipping {f}: {e}")
            continue

    if not top_hits:
        print("   ❌ No salt found.")
        return

    top_hits.sort(key=lambda x: x['score'], reverse=True)
    top_hits = top_hits[:3]

    print(f"   🎨 Saving Report...")
    n_plots = len(top_hits)
    fig, axs = plt.subplots(1, n_plots, figsize=(6 * n_plots, 18))
    if n_plots == 1: axs = [axs]
    
    for i, ex in enumerate(top_hits):
        ax = axs[i]
        seismic = ex['seismic']
        mask = ex['mask']
        
        extent = [0, seismic.shape[1], seismic.shape[0], 0]
        
        ax.imshow(seismic, cmap='gray', aspect='auto', extent=extent)
        
        overlay = np.zeros((mask.shape[0], mask.shape[1], 4))
        overlay[mask == 1] = [1, 0.9, 0, 0.5] 
        ax.imshow(overlay, aspect='auto', extent=extent)
        
        ax.set_title(f"{ex['file']}\nInline: {ex['inline']}", fontsize=10, fontweight='bold')
        ax.set_xlabel("Trace Number", fontsize=9)
        if i == 0: ax.set_ylabel("Depth Sample", fontsize=9)
        
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=150)
    print(f"   📸 DONE: {OUTPUT_IMAGE}")

if __name__ == "__main__":
    main()
