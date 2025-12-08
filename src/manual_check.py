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
OUTPUT_IMAGE = "Manual_Check_Report.png"
MODEL_PATH = "models/best_model.pth"

# PHYSICS CORRECTION (For Model Resampling)
SCALE_FACTOR = 0.4  # Resample 4ms -> 10ms

# --- 1. DEFINE THE FILE'S FULL RANGE ---
# Enter the Top and Bottom values exactly as seen in Petrel for the WHOLE file.
# This establishes the "Math" to convert units to indices.
# Example: If your file goes from 0 to -12000ms, enter (0, -12000)
FILE_FULL_RANGE = (-12, -11912) 

# --- 2. DEFINE THE CROP WINDOW ---
# Where do you want to slice? (Must be inside the full range above)
# Example: I only want to check the top section (-12 to -7000)
CROP_WINDOW = (-12, -7000)

# SPECIFIC INLINES TO CHECK
TARGETS = {
    "G3D202407-01_lns893_1582.sgy": [1227],
}
# ======================================================

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
        # === DEBUG / CROP LOGIC ===
        print("\n   --- 🛠️ CROP CALCULATOR ---")
        n_samples = slice_data.shape[0]
        
        # 1. Calculate Units Per Sample based on User's Full Range
        file_top, file_bot = FILE_FULL_RANGE
        total_span = abs(file_bot - file_top)
        
        if n_samples > 0:
            units_per_sample = total_span / n_samples
        else:
            units_per_sample = 0
            
        print(f"   > File Full Range: {FILE_FULL_RANGE} ({total_span} units)")
        print(f"   > Trace Length: {n_samples} samples")
        print(f"   > Calculated Resolution: {units_per_sample:.4f} units/sample")

        # 2. Convert Crop Window to Indices
        crop_top, crop_bot = CROP_WINDOW
        
        # Shift values relative to file start (normalize to 0)
        # We assume 'top' is the start, regardless of sign (e.g. 0 or -12000)
        # Distance from Start / Resolution = Index
        
        # Determine strict numeric range (Handling negative coordinates)
        # We map everything to distance from the "Top" (Start of file)
        
        # Distance of crop_top from file_top
        dist_start = abs(crop_top - file_top)
        # Distance of crop_bot from file_top
        dist_end = abs(crop_bot - file_top)
        
        idx_start = int(dist_start / units_per_sample)
        idx_end = int(dist_end / units_per_sample)
        
        # Ensure Start < End for slicing
        if idx_start > idx_end: idx_start, idx_end = idx_end, idx_start
        
        print(f"   > User Crop Request: {CROP_WINDOW}")
        print(f"   > Converting to Indices: [{idx_start}:{idx_end}]")
        
        # 3. Apply Crop
        idx_start = max(0, idx_start)
        idx_end = min(n_samples, idx_end)
        
        slice_data = slice_data[idx_start:idx_end, :]
        print(f"   > Final Crop Shape: {slice_data.shape}")
        # ==========================

        # 4. Standard Processing
        slice_resampled = zoom(slice_data, (SCALE_FACTOR, 1.0), order=1)
        
        h_new = slice_resampled.shape[0]
        if h_new > 1024:
            input_tensor_data = slice_resampled[:1024, :]
        elif h_new < 1024:
            pad = np.zeros((1024, slice_resampled.shape[1]))
            pad[:h_new, :] = slice_resampled
            input_tensor_data = pad
        else:
            input_tensor_data = slice_resampled
            
        h, w = input_tensor_data.shape
        target_w = ((w + 31) // 32) * 32
        pad_w = target_w - w
        if pad_w > 0: input_tensor_data = np.pad(input_tensor_data, ((0,0), (0, pad_w)), mode='constant')
            
        slice_norm = robust_normalize_tile(input_tensor_data)
        tensor = torch.from_numpy(slice_norm).float().unsqueeze(0).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            out = self.model(tensor)
            prob = torch.softmax(out, dim=1)
            salt_prob = prob[0, 2, :, :].cpu().numpy()
            
        salt_prob = salt_prob[:, :w]
        
        depth_ramp = np.linspace(0.5, 0.95, h).reshape(-1, 1)
        dynamic_threshold = np.tile(depth_ramp, (1, w))
        binary = salt_prob > dynamic_threshold
        clean = morphology.remove_small_objects(binary, min_size=200)
        
        final_mask = clean
        if h_new < 1024: final_mask = final_mask[:h_new, :]
            
        final_mask_orig = zoom(final_mask, (1/SCALE_FACTOR, 1.0), order=0)
        
        final_h = min(slice_data.shape[0], final_mask_orig.shape[0])
        final_mask_orig = final_mask_orig[:final_h, :]
        slice_data = slice_data[:final_h, :]

        return final_mask_orig, slice_data 

def get_specific_inline(src, target_il):
    n_traces = src.tracecount
    try:
        start_il = src.header[0][segyio.TraceField.INLINE_3D]
        end_il = src.header[n_traces-1][segyio.TraceField.INLINE_3D]
        
        if not (min(start_il, end_il) <= target_il <= max(start_il, end_il)):
            print(f"      ❌ Inline {target_il} is outside file range ({start_il}-{end_il})")
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
            print("      ⚠️ Math jump missed. Falling back to linear scan...")
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

def main():
    if not os.path.exists(INPUT_FOLDER): 
        print(f"❌ Input folder not found: {INPUT_FOLDER}")
        return
        
    hunter = SeismicHunter(MODEL_PATH)
    results = []
    
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
                        mask_pred, seismic_crop = hunter.predict_slice(slice_data)
                        results.append({
                            'file': filename,
                            'inline': target_il,
                            'seismic': seismic_crop,
                            'mask': mask_pred
                        })
                    else:
                        print(f"      ❌ Inline {target_il} not found.")
        except Exception as e:
            print(f"   ⚠️ Error reading {filename}: {e}")
            continue

    if not results:
        print("   ❌ No data captured.")
        return

    # === PLOTTING ===
    print(f"   🎨 Generating Report...")
    n_plots = len(results)
    fig, axs = plt.subplots(1, n_plots, figsize=(18 * n_plots, 10))
    if n_plots == 1: axs = [axs]
    
    for i, ex in enumerate(results):
        ax = axs[i]
        seismic = ex['seismic']
        mask = ex['mask']
        
        # EXTENT MAPPING
        # Now we map the pixels to the exact CROP WINDOW the user asked for
        if CROP_WINDOW:
            z_vals = sorted([CROP_WINDOW[0], CROP_WINDOW[1]], reverse=True)
            extent = [0, seismic.shape[1], z_vals[1], z_vals[0]]
        else:
            extent = [0, seismic.shape[1], seismic.shape[0], 0]
        
        ax.imshow(seismic, cmap='gray', aspect='auto', extent=extent)
        
        overlay = np.zeros((mask.shape[0], mask.shape[1], 4))
        overlay[mask == 1] = [1, 0.9, 0, 0.5] 
        ax.imshow(overlay, aspect='auto', extent=extent)
        
        ax.set_title(f"{ex['file']}\nInline: {ex['inline']}", fontsize=14, fontweight='bold')
        ax.set_xlabel("Trace Number", fontsize=12)
        if i == 0: ax.set_ylabel("Depth/Time (Petrel Units)", fontsize=12)
        ax.grid(color='cyan', linestyle='-', linewidth=0.3, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=150)
    print(f"   📸 Report Saved: {OUTPUT_IMAGE}")

if __name__ == "__main__":
    main()
