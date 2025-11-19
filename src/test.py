import segyio
import numpy as np
import matplotlib.pyplot as plt
import os

# ================= CONFIGURATION =================
LABEL_FILENAME = "SaltMask.segy"
TARGET_INLINE = 14000
# =================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILE_PATH = os.path.join(BASE_DIR, 'data', 'raw', LABEL_FILENAME)

def check_specific_inline():
    if not os.path.exists(FILE_PATH):
        print(f"ERROR: Could not find {FILE_PATH}")
        return

    print(f"Opening {LABEL_FILENAME}...")
    
    with segyio.open(FILE_PATH, ignore_geometry=False, strict=False) as f:
        if TARGET_INLINE not in f.ilines:
            print(f"ERROR: Inline {TARGET_INLINE} is not in this file.")
            return

        print(f"Reading Inline {TARGET_INLINE}...")
        slice_data = f.iline[TARGET_INLINE]
        
        # --- FIX 1: Handle Floating Point Artifacts ---
        # 1.0039216 -> 1
        print(f"Original Raw Values: {np.unique(slice_data)}")
        slice_data = np.round(slice_data).astype(int)
        print(f"Corrected Integer Values: {np.unique(slice_data)}")
        
        # 4. Visualization
        plt.figure(figsize=(12, 8))
        
        h, w = slice_data.shape
        rgb_image = np.zeros((h, w, 3))
        
        # Colors: 0=Gray, 1=Cyan, 4=Yellow
        rgb_image[slice_data == 0] = [0.2, 0.2, 0.2] 
        rgb_image[slice_data == 1] = [0.0, 1.0, 1.0] 
        rgb_image[slice_data == 4] = [1.0, 1.0, 0.0]

        # Check for weird values
        unexpected_mask = ~np.isin(slice_data, [0, 1, 4])
        if np.any(unexpected_mask):
            rgb_image[unexpected_mask] = [1.0, 0.0, 0.0] # Red for error

        # --- FIX 2: Flip to 'upper' (Geological Depth) ---
        # origin='upper' puts Sample 0 (Water Surface) at the TOP of the image
        plt.imshow(rgb_image.transpose(1, 0, 2), aspect='auto', origin='upper')
        
        plt.title(f"Inline {TARGET_INLINE} (Corrected)\nTop=Surface | Cyan=Water | Yellow=Salt")
        plt.xlabel("Crossline Index")
        plt.ylabel("Depth (Sample)")
        
        save_path = os.path.join(BASE_DIR, 'data', 'processed', f'inline_{TARGET_INLINE}_fixed.png')
        plt.savefig(save_path)
        print(f"Fixed profile saved to: {save_path}")

if __name__ == "__main__":
    check_specific_inline()