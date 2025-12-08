import os
import torch
import numpy as np
import segyio
import pandas as pd
from skimage import morphology
from skimage.morphology import disk
from tqdm import tqdm
from model import UNet

# ================= USER CONFIGURATION =================
INPUT_FOLDER = r"G:\Working\Students\Undergraduate\For_Vince\SeismicFacies\namss.B-26-99-TX.mcs3d.airgun\Data\segy"
OUTPUT_FILE = os.path.join(INPUT_FOLDER, "Combined_Salt_Model_v1.csv")

MODEL_PATH = "models/best_model.pth"

# GRID SETTINGS (Speed vs. Density)
# Process every Nth Inline/Crossline.
# 1 = Full resolution (Slowest, Heaviest file)
# 5 = High Res Grid (Recommended)
# 10 = Quick Look
STEP_INLINE = 20    
STEP_CROSSLINE = 1  # Usually keep XL step at 1 if iterating Inlines to get full slices

# TILE SETTINGS
PATCH_HEIGHT = 1024
PATCH_WIDTH = 256
OVERLAP_H = 128
OVERLAP_W = 64
# ======================================================

def robust_normalize_tile(img):
    """ Matches training normalization. """
    mean = np.mean(img)
    std = np.std(img)
    lower = mean - 2.5 * std
    upper = mean + 2.5 * std
    img = np.clip(img, lower, upper)
    min_val = np.min(img)
    max_val = np.max(img)
    if max_val - min_val == 0: return np.zeros_like(img)
    img = 2 * (img - min_val) / (max_val - min_val) - 1.0
    return img

class SeismicPredictor:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"   🚀 Loading Model from {model_path}...")
        self.model = UNet(n_channels=1, n_classes=3).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def predict_large_slice(self, full_slice):
        """ Returns the Cleaned Binary Salt Mask (0 or 1) """
        h_full, w_full = full_slice.shape
        probs_map = np.zeros((h_full, w_full), dtype=np.float32)
        counts_map = np.zeros((h_full, w_full), dtype=np.float32)

        step_h = PATCH_HEIGHT - OVERLAP_H
        step_w = PATCH_WIDTH - OVERLAP_W

        for y in range(0, h_full, step_h):
            for x in range(0, w_full, step_w):
                y_end = min(y + PATCH_HEIGHT, h_full)
                x_end = min(x + PATCH_WIDTH, w_full)
                y_start = max(0, y_end - PATCH_HEIGHT)
                x_start = max(0, x_end - PATCH_WIDTH)

                tile = full_slice[y_start:y_end, x_start:x_end]
                tile_norm = robust_normalize_tile(tile)
                tensor = torch.from_numpy(tile_norm).float().unsqueeze(0).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    out = self.model(tensor)
                    prob = torch.softmax(out, dim=1)
                    salt_tile = prob[0, 2, :, :].cpu().numpy()

                probs_map[y_start:y_end, x_start:x_end] += salt_tile
                counts_map[y_start:y_end, x_start:x_end] += 1

        final_probs = probs_map / (counts_map + 1e-6)
        
        # Cleanup
        binary = final_probs > 0.5
        clean = morphology.remove_small_objects(binary, min_size=64)
        clean = morphology.binary_closing(clean, disk(3))
        
        return clean # Returns Boolean Mask (True where Salt)

def process_and_stream(files, predictor, output_csv):
    """ Loops through files and streams salt points to CSV immediately """
    
    # 1. Create CSV with Headers
    # We will write mode='w' (overwrite) first, then append.
    headers = "Inline,Crossline,Sample_Z,Salt_Prob\n"
    with open(output_csv, "w") as f:
        f.write(headers)
    
    print(f"   📄 Created Master CSV: {output_csv}")

    total_points = 0
    
    for filepath in files:
        filename = os.path.basename(filepath)
        print(f"\n📂 Processing File: {filename}")
        
        try:
            with segyio.open(filepath, "r", ignore_geometry=True) as src:
                src.mmap()
                
                # Get Geometry
                inlines = src.attributes(segyio.TraceField.INLINE_3D)[:]
                crosslines = src.attributes(segyio.TraceField.CROSSLINE_3D)[:]
                unique_ils = np.unique(inlines)
                
                print(f"   Grid: Processing every {STEP_INLINE}th Inline...")
                target_ils = unique_ils[::STEP_INLINE]
                
                # Buffer for batch writing to CSV (faster than line-by-line)
                csv_buffer = []
                BATCH_SIZE = 100000

                for il in tqdm(target_ils, desc=f"Scanning {filename}"):
                    mask_il = (inlines == il)
                    trace_indices = np.where(mask_il)[0]
                    if len(trace_indices) == 0: continue
                    
                    # Get Slice
                    # slice_data shape: (Num_XL, Samples)
                    slice_data = src.trace[trace_indices]
                    
                    # Get Corresponding Crossline Numbers for these traces
                    xl_nums = crosslines[trace_indices]
                    
                    # Predict (Needs Transpose -> Samples, XL)
                    pred_mask = predictor.predict_large_slice(slice_data.T)
                    
                    # pred_mask is (Samples, XL). We need to map back to coordinates.
                    # Find indices where Salt is TRUE
                    # z_indices (Samples), x_indices (index in xl_nums)
                    salt_locs = np.where(pred_mask) 
                    
                    if len(salt_locs[0]) > 0:
                        z_inds = salt_locs[0]
                        xl_inds = salt_locs[1]
                        
                        # Map internal array index back to Real Crossline Number
                        real_xls = xl_nums[xl_inds]
                        
                        # Add to buffer: IL, XL, Z, Prob(1)
                        # We stack them into rows
                        for z, xl in zip(z_inds, real_xls):
                            # Format: IL, XL, Z, Prob
                            # Z is usually Sample Index. If you want Time/Depth, multiply by sample rate.
                            # Here we assume Sample Index (0, 1, 2...)
                            csv_buffer.append(f"{il},{xl},{z},1.0\n")
                            
                            if len(csv_buffer) >= BATCH_SIZE:
                                with open(output_csv, "a") as f:
                                    f.writelines(csv_buffer)
                                total_points += len(csv_buffer)
                                csv_buffer = []

                # Flush remaining points for this file
                if csv_buffer:
                    with open(output_csv, "a") as f:
                        f.writelines(csv_buffer)
                    total_points += len(csv_buffer)

        except Exception as e:
            print(f"   ⚠️ Error reading {filename}: {e}")
            continue

    print(f"\n🎉 DONE! Total Salt Points extracted: {total_points}")
    print(f"   Import '{output_csv}' into Petrel as 'Point Set'.")

def main():
    if not os.path.exists(INPUT_FOLDER):
        print("❌ Input folder not found.")
        return

    predictor = SeismicPredictor(MODEL_PATH)
    
    files = [os.path.join(INPUT_FOLDER, f) for f in os.listdir(INPUT_FOLDER) if f.endswith('.segy') or f.endswith('.sgy')]
    print(f"Found {len(files)} SEGY files to merge.")
    
    process_and_stream(files, predictor, OUTPUT_FILE)

if __name__ == "__main__":
    main()
