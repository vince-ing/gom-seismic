import os
import torch
import numpy as np
import segyio
import matplotlib.pyplot as plt
from skimage import morphology
from scipy.ndimage import zoom
from model import UNet

# ================= USER CONFIGURATION =================
# INPUTS
INPUT_FOLDER = r"G:\Working\Students\Undergraduate\For_Vince\SeismicFacies\gom-seismic\data\raw"
TARGET_FILE = "CroppedData.segy" 

# OPTIONAL COMPARISON
# Set to "SaltMask.segy" to overlay Ground Truth on the top plot.
# Set to "" (empty string) to just plot Seismic on top.
MASK_FILE = "SaltMask.segy"  

# OUTPUT
OUTPUT_FOLDER = r"G:\Working\Students\Undergraduate\For_Vince\SeismicFacies\gom-seismic\data\processed\originalcheck"
MODEL_PATH = "models/best_model.pth"

# --- STEPPING LOGIC ---
START_INLINE = 14000  
STEP_SIZE = 50        
NUM_STEPS = 25         

# PHYSICS & CROP
SCALE_FACTOR = 1.0  # Crisp Data (No Resampling)

# Headers/Cropping
FILE_FULL_RANGE = (0, 100) 
CROP_WINDOW = None

# VISUAL TUNING
SIGMA_CLIP_MODEL = 2.5 
VISUAL_SEISMIC_CLIP = 0.3 
VISUAL_GAMMA_PROB = 0.5 
# ======================================================

def robust_normalize_tile(img):
    mean = np.mean(img)
    std = np.std(img)
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

    def process_geometry_only(self, slice_data, is_mask=False):
        """
        Applies Crop, Resample, and Pad logic EXACTLY same as prediction.
        """
        # 1. CROP
        n_samples = slice_data.shape[0]
        if FILE_FULL_RANGE and CROP_WINDOW:
            file_top, file_bot = FILE_FULL_RANGE
            units_per_sample = abs(file_bot - file_top) / n_samples if n_samples > 0 else 0
            
            crop_top, crop_bot = CROP_WINDOW
            idx_start = int(abs(crop_top - file_top) / units_per_sample)
            idx_end = int(abs(crop_bot - file_top) / units_per_sample)
            if idx_start > idx_end: idx_start, idx_end = idx_end, idx_start
            idx_start = max(0, idx_start)
            idx_end = min(n_samples, idx_end)
            slice_data = slice_data[idx_start:idx_end, :]

        # 2. RESAMPLE
        if SCALE_FACTOR != 1.0:
            order = 0 if is_mask else 1
            slice_resampled = zoom(slice_data, (SCALE_FACTOR, 1.0), order=order)
        else:
            slice_resampled = slice_data

        # 3. PAD TO 1024
        h_new = slice_resampled.shape[0]
        if h_new > 1024:
            processed = slice_resampled[:1024, :]
        elif h_new < 1024:
            pad = np.zeros((1024, slice_resampled.shape[1]))
            pad[:h_new, :] = slice_resampled
            processed = pad
        else:
            processed = slice_resampled
            
        # 4. PAD WIDTH
        h, w = processed.shape
        target_w = ((w + 31) // 32) * 32
        pad_w = target_w - w
        if pad_w > 0: processed = np.pad(processed, ((0,0), (0, pad_w)), mode='constant')

        return processed, h_new, w, slice_data.shape[0]

    def predict_slice(self, slice_data):
        # Apply Geometry Transforms
        input_data, h_new, w, original_h = self.process_geometry_only(slice_data, is_mask=False)
        
        # Normalize (Model Only)
        slice_norm = robust_normalize_tile(input_data)
        
        # Predict
        tensor = torch.from_numpy(slice_norm).float().unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(tensor)
            prob = torch.softmax(out, dim=1)
            salt_prob = prob[0, 2, :, :].cpu().numpy()
            
        # Cleanup
        salt_prob = salt_prob[:, :w]
        if h_new < 1024: salt_prob = salt_prob[:h_new, :]
        
        if SCALE_FACTOR != 1.0:
            final_prob = zoom(salt_prob, (1/SCALE_FACTOR, 1.0), order=1)
        else:
            final_prob = salt_prob
            
        final_h = min(original_h, final_prob.shape[0])
        return final_prob[:final_h, :]

    def prepare_ground_truth(self, mask_data):
        """
        Processes the mask file to align with the seismic plot.
        """
        processed, h_new, w, original_h = self.process_geometry_only(mask_data, is_mask=True)
        
        processed = processed[:, :w]
        if h_new < 1024: processed = processed[:h_new, :]
        
        if SCALE_FACTOR != 1.0:
            final_mask = zoom(processed, (1/SCALE_FACTOR, 1.0), order=0)
        else:
            final_mask = processed
            
        final_h = min(original_h, final_mask.shape[0])
        return final_mask[:final_h, :]

def get_specific_inline(src, target_il):
    n_traces = src.tracecount
    try:
        start_il = src.header[0][segyio.TraceField.INLINE_3D]
        end_il = src.header[n_traces-1][segyio.TraceField.INLINE_3D]
        
        if not (min(start_il, end_il) <= target_il <= max(start_il, end_il)): return None

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
    except Exception: return None

def plot_and_save(filename, inline, seismic, prob, mask_gt=None):
    if not os.path.exists(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER)
    
    # 1. NAMING LOGIC
    clean_name = os.path.splitext(filename)[0]
    
    if mask_gt is not None:
        suffix = "_Comp"  # Has comparison
    else:
        suffix = ""       # Standard check
        
    save_path = os.path.join(OUTPUT_FOLDER, f"{clean_name}_Inline_{inline}{suffix}.png")
    
    # Force seismic geometry (ensure prediction matches exactly)
    min_h = min(seismic.shape[0], prob.shape[0])
    if mask_gt is not None: min_h = min(min_h, mask_gt.shape[0])
    
    seismic = seismic[:min_h, :]
    prob = prob[:min_h, :]
    if mask_gt is not None: mask_gt = mask_gt[:min_h, :]

    # Setup Plot
    fig, axs = plt.subplots(2, 1, figsize=(20, 16), sharex=True)
    
    if CROP_WINDOW:
        z_vals = sorted([CROP_WINDOW[0], CROP_WINDOW[1]], reverse=True)
        extent = [0, seismic.shape[1], z_vals[1], z_vals[0]]
    else:
        extent = [0, seismic.shape[1], seismic.shape[0], 0]

    # --- TOP PLOT ---
    ax1 = axs[0]
    ax1.imshow(seismic, cmap='gray', aspect='auto', extent=extent, 
               vmin=-VISUAL_SEISMIC_CLIP, vmax=VISUAL_SEISMIC_CLIP)
    
    if mask_gt is not None:
        # Mask 0s, show 1s as Gold
        mask_overlay = np.ma.masked_where(mask_gt < 0.5, mask_gt)
        ax1.imshow(mask_overlay, cmap='viridis', aspect='auto', extent=extent, alpha=0.6)
        title_extra = "| Ground Truth (Gold)"
    else:
        title_extra = "| Seismic Only"

    ax1.set_title(f"{filename} | Inline {inline} {title_extra}", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Depth/Time (Petrel Units)", fontsize=12)
    ax1.grid(color='cyan', linestyle='-', linewidth=0.3, alpha=0.3)

    # --- BOTTOM PLOT ---
    ax2 = axs[1]
    ax2.imshow(seismic, cmap='gray', aspect='auto', extent=extent, 
               vmin=-VISUAL_SEISMIC_CLIP, vmax=VISUAL_SEISMIC_CLIP)
    
    prob_boosted = np.power(prob, VISUAL_GAMMA_PROB)
    prob_masked = np.ma.masked_where(prob_boosted < 0.05, prob_boosted)
    
    ax2.imshow(prob_masked, cmap='jet', aspect='auto', extent=extent, alpha=0.5, vmin=0.0, vmax=1.0)
    
    ax2.set_title(f"Model Prediction (Jet)", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Trace Number", fontsize=12)
    ax2.set_ylabel("Depth/Time (Petrel Units)", fontsize=12)
    ax2.grid(color='white', linestyle='--', linewidth=0.5, alpha=0.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"   📸 Saved: {save_path}")

def main():
    if not os.path.exists(INPUT_FOLDER): 
        print(f"❌ Input folder not found: {INPUT_FOLDER}")
        return
        
    hunter = SeismicHunter(MODEL_PATH)
    file_seis = os.path.join(INPUT_FOLDER, TARGET_FILE)
    
    # Optional Mask Logic
    if MASK_FILE:
        file_mask = os.path.join(INPUT_FOLDER, MASK_FILE)
        has_mask = os.path.exists(file_mask)
    else:
        file_mask = None
        has_mask = False
        
    if not os.path.exists(file_seis):
        print(f"❌ Seismic file not found: {TARGET_FILE}")
        return

    if has_mask:
        print(f"   ✅ Mask found. Comparison Mode ON.")
    else:
        print(f"   ℹ️  No mask provided/found. Standard Mode.")

    print(f"   📂 Analyzing: {TARGET_FILE}")

    try:
        with segyio.open(file_seis, "r", ignore_geometry=True) as src_seis:
            # Open Mask conditionally
            src_mask = segyio.open(file_mask, "r", ignore_geometry=True) if has_mask else None
            
            try:
                n_traces = src_seis.tracecount
                start_il = src_seis.header[0][segyio.TraceField.INLINE_3D]
                end_il = src_seis.header[n_traces-1][segyio.TraceField.INLINE_3D]
                il_min, il_max = min(start_il, end_il), max(start_il, end_il)
                
                current_il = START_INLINE if START_INLINE is not None else il_min
                targets = []
                for i in range(NUM_STEPS):
                    target = current_il + (i * STEP_SIZE)
                    if target < il_min or target > il_max: continue
                    targets.append(target)
                
                print(f"   🎯 Processing {len(targets)} targets...")

                for target_il in targets:
                    print(f"\n   ...processing Inline {target_il}")
                    slice_seis = get_specific_inline(src_seis, target_il)
                    
                    if slice_seis is not None:
                        prob_map = hunter.predict_slice(slice_seis)
                        
                        mask_gt = None
                        if src_mask:
                            # Try to get matching mask line
                            slice_mask = get_specific_inline(src_mask, target_il)
                            if slice_mask is not None:
                                mask_gt = hunter.prepare_ground_truth(slice_mask)
                            else:
                                print(f"      ⚠️ Mask missing for Inline {target_il}")

                        # Get display crop
                        seismic_crop, _, _, _ = hunter.process_geometry_only(slice_seis, is_mask=False)
                        seismic_crop = seismic_crop[:prob_map.shape[0], :prob_map.shape[1]]

                        plot_and_save(TARGET_FILE, target_il, seismic_crop, prob_map, mask_gt)
                    else:
                        print(f"      ❌ Failed to capture Inline {target_il}")

            finally:
                if src_mask: src_mask.close()

    except Exception as e:
        print(f"   ⚠️ Error: {e}")

if __name__ == "__main__":
    main()
