import os
import torch
import numpy as np
import segyio
import matplotlib.pyplot as plt
from skimage import morphology
from scipy.ndimage import zoom
from model import UNet

# ================= USER CONFIGURATION =================
INPUT_FOLDER = r"G:\Working\Students\Undergraduate\For_Vince\SeismicFacies\namss.B-26-99-TX.mcs3d.airgun\Data\segy"
OUTPUT_FOLDER = r"G:\Working\Students\Undergraduate\For_Vince\SeismicFacies\gom-seismic\data\processed\lines"
MODEL_PATH = "models/best_model.pth"

# PHYSICS CORRECTION
SCALE_FACTOR = 0.4  # Resample 4ms -> 10ms

# --- 1. DEFINE THE FILE'S FULL RANGE ---
FILE_FULL_RANGE = (-12, -11912) 

# --- 2. DEFINE THE CROP WINDOW ---
CROP_WINDOW = (-12, -7000)

# --- 3. TUNING CONTROLS ---
# MODEL INPUT: Controls what the AI sees (Standard=2.5)
SIGMA_CLIP_MODEL = 2.5 

# VISUAL OUTPUT: Controls the report appearance
# 1.0 = Standard. 0.3 = High Contrast (Triples the visible contrast).
VISUAL_SEISMIC_CLIP = 0.3 
VISUAL_GAMMA_PROB = 0.5 

# SPECIFIC INLINES TO CHECK
TARGETS = {
    #"G3D202407-01_lns893_1582.sgy": [1227],
    "G3D202407-01_lns1583_2272.sgy": [1968, 1938, 1928, 1918],
    # Add more here
}
# ======================================================

def robust_normalize_tile(img):
    """
    Standard Normalization for the AI Model.
    """
    mean = np.mean(img)
    std = np.std(img)
    
    # Clip based on model config (Standard 2.5)
    img = np.clip(img, mean - SIGMA_CLIP_MODEL * std, mean + SIGMA_CLIP_MODEL * std)
    
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
        print("\n   --- 🛠️ PROCESSING PIPELINE ---")
        
        # 1. CROP
        n_samples = slice_data.shape[0]
        file_top, file_bot = FILE_FULL_RANGE
        units_per_sample = abs(file_bot - file_top) / n_samples if n_samples > 0 else 0
        
        crop_top, crop_bot = CROP_WINDOW
        idx_start = int(abs(crop_top - file_top) / units_per_sample)
        idx_end = int(abs(crop_bot - file_top) / units_per_sample)
        if idx_start > idx_end: idx_start, idx_end = idx_end, idx_start
        
        slice_data = slice_data[idx_start:idx_end, :]
        print(f"   > 1. Cropped to: {slice_data.shape}")

        # 2. RESAMPLE
        slice_resampled = zoom(slice_data, (SCALE_FACTOR, 1.0), order=1)
        print(f"   > 2. Resampled to: {slice_resampled.shape}")
        
        # 3. NORMALIZE (For Model)
        slice_norm = robust_normalize_tile(slice_resampled)

        # 4. PAD
        h_new = slice_norm.shape[0]
        if h_new > 1024:
            input_tensor_data = slice_norm[:1024, :]
        elif h_new < 1024:
            pad = np.zeros((1024, slice_norm.shape[1]))
            pad[:h_new, :] = slice_norm
            input_tensor_data = pad
            print(f"   > 3. Padded {1024-h_new} pixels")
        else:
            input_tensor_data = slice_norm
            
        # 5. PAD WIDTH
        h, w = input_tensor_data.shape
        target_w = ((w + 31) // 32) * 32
        pad_w = target_w - w
        if pad_w > 0: input_tensor_data = np.pad(input_tensor_data, ((0,0), (0, pad_w)), mode='constant')
            
        # 6. PREDICT
        tensor = torch.from_numpy(input_tensor_data).float().unsqueeze(0).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            out = self.model(tensor)
            prob = torch.softmax(out, dim=1)
            salt_prob = prob[0, 2, :, :].cpu().numpy()
            
        # 7. CLEANUP
        salt_prob = salt_prob[:, :w]
        
        final_prob = salt_prob
        if h_new < 1024: final_prob = final_prob[:h_new, :]
            
        final_prob_orig = zoom(final_prob, (1/SCALE_FACTOR, 1.0), order=1)
        
        final_h = min(slice_data.shape[0], final_prob_orig.shape[0])
        final_prob_orig = final_prob_orig[:final_h, :]
        slice_data = slice_data[:final_h, :]

        return final_prob_orig, slice_data 

def get_specific_inline(src, target_il):
    n_traces = src.tracecount
    try:
        start_il = src.header[0][segyio.TraceField.INLINE_3D]
        end_il = src.header[n_traces-1][segyio.TraceField.INLINE_3D]
        
        if not (min(start_il, end_il) <= target_il <= max(start_il, end_il)):
            return None

        if end_il - start_il == 0: fraction = 0
        else: fraction = (target_il - start_il) / (end_il - start_il)
        
        approx_idx = int(fraction * n_traces)
        search_radius = 20000 
        start_search = max(0, approx_idx - search_radius)
        end_search = min(n_traces, approx_idx + search_radius)
        
        chunk_headers = src.header[start_search:end_search]
        chunk_ils = [h[segyio.TraceField.INLINE_3D] for h in chunk_headers]
        
        chunk_ils_arr = np.array(chunk_ils)
        mask = (chunk_ils_arr == target_il)
        rel_indices = np.where(mask)[0]
        
        if len(rel_indices) > 0:
            abs_start = start_search + rel_indices[0]
            abs_end = start_search + rel_indices[-1] + 1
            print(f"      ✅ Jumped to trace {abs_start} (Hit!)")
            return np.stack([src.trace[i] for i in range(abs_start, abs_end)]).T
        else:
            current_idx = 0
            while current_idx < n_traces:
                end_chunk = min(current_idx + 20000, n_traces)
                chunk_headers = src.header[current_idx:end_chunk]
                chunk_ils = [h[segyio.TraceField.INLINE_3D] for h in chunk_headers]
                if target_il in chunk_ils:
                    mask = (np.array(chunk_ils) == target_il)
                    rel_indices = np.where(mask)[0]
                    abs_start = current_idx + rel_indices[0]
                    abs_end = current_idx + rel_indices[-1] + 1
                    return np.stack([src.trace[i] for i in range(abs_start, abs_end)]).T
                current_idx += 20000
            return None
    except Exception as e:
        print(f"      ⚠️ Seek Error: {e}")
        return None

def plot_and_save(filename, inline, seismic, prob):
    """
    Saves a dual-plot report (Seismic Only / Seismic + Prob).
    """
    # Create Output Directory
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    # Clean Filename
    clean_name = os.path.splitext(filename)[0]
    save_path = os.path.join(OUTPUT_FOLDER, f"{clean_name}_Inline_{inline}.png")
    
    # Setup Figure (2 Rows, 1 Column)
    fig, axs = plt.subplots(2, 1, figsize=(20, 16))
    
    # Extent Mapping
    if CROP_WINDOW:
        z_vals = sorted([CROP_WINDOW[0], CROP_WINDOW[1]], reverse=True)
        extent = [0, seismic.shape[1], z_vals[1], z_vals[0]]
    else:
        extent = [0, seismic.shape[1], seismic.shape[0], 0]

    # --- TOP PLOT: SEISMIC ONLY (High Contrast) ---
    ax1 = axs[0]
    # We deliberately clip visual display to +/- 0.3 for high contrast
    # (Data remains -1 to 1)
    ax1.imshow(seismic, cmap='gray', aspect='auto', extent=extent, 
               vmin=-VISUAL_SEISMIC_CLIP, vmax=VISUAL_SEISMIC_CLIP)
    
    ax1.set_title(f"{filename} | Inline {inline} | Seismic Only", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Depth/Time (Petrel Units)", fontsize=12)
    ax1.grid(color='cyan', linestyle='-', linewidth=0.3, alpha=0.3)

    # --- BOTTOM PLOT: PREDICTION OVERLAY ---
    ax2 = axs[1]
    ax2.imshow(seismic, cmap='gray', aspect='auto', extent=extent, 
               vmin=-VISUAL_SEISMIC_CLIP, vmax=VISUAL_SEISMIC_CLIP)
    
    # Gamma Correct Probability
    prob_boosted = np.power(prob, VISUAL_GAMMA_PROB)
    prob_masked = np.ma.masked_where(prob_boosted < 0.05, prob_boosted)
    
    im = ax2.imshow(prob_masked, cmap='jet', aspect='auto', extent=extent, alpha=0.5, vmin=0.0, vmax=1.0)
    
    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label(f'Salt Probability (Gamma={VISUAL_GAMMA_PROB})', rotation=270, labelpad=15)
    
    ax2.set_title(f"Salt Prediction Overlay", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Trace Number", fontsize=12)
    ax2.set_ylabel("Depth/Time (Petrel Units)", fontsize=12)
    ax2.grid(color='white', linestyle='--', linewidth=0.5, alpha=0.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig) # Close memory
    print(f"   📸 Saved: {save_path}")

def main():
    if not os.path.exists(INPUT_FOLDER): 
        print(f"❌ Input folder not found: {INPUT_FOLDER}")
        return
        
    hunter = SeismicHunter(MODEL_PATH)
    
    print(f"   🔎 MANUAL MODE: Checking specific inlines...")

    for filename, inline_list in TARGETS.items():
        filepath = os.path.join(INPUT_FOLDER, filename)
        if not os.path.exists(filepath):
            print(f"   ⚠️ File not found: {filename}")
            continue
            
        print(f"   📂 Processing {filename}...")
        try:
            with segyio.open(filepath, "r", ignore_geometry=True) as src:
                for target_il in inline_list:
                    print(f"      ...seeking Inline {target_il}")
                    slice_data = get_specific_inline(src, target_il)
                    
                    if slice_data is not None:
                        prob_map, seismic_crop = hunter.predict_slice(slice_data)
                        
                        # Plot Immediately per inline
                        plot_and_save(filename, target_il, seismic_crop, prob_map)
                        
                    else:
                        print(f"      ❌ Inline {target_il} not found.")
        except Exception as e:
            print(f"   ⚠️ Error reading {filename}: {e}")
            continue

if __name__ == "__main__":
    main()
