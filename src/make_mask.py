import segyio
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import datetime

# ================= CONFIGURATION =================
LABEL_FILENAME = "SaltMask.segy" 
STEP = 10   # Keep 10 for preview, change to 1 for final run
VALID_VALUES = [1, 4] # Water and Salt
# =================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, 'data', 'raw', LABEL_FILENAME)
OUTPUT_MASK_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'survey_mask.npy')
OUTPUT_IMAGE_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'survey_mask_preview.png')

def generate_mask_fast():
    if not os.path.exists(INPUT_PATH):
        print(f"ERROR: File not found at {INPUT_PATH}")
        return

    print(f"{'='*40}")
    print(f"Starting Mask Generation (Fix applied: Rounding values)")
    print(f"{'='*40}")

    with segyio.open(INPUT_PATH, ignore_geometry=False, strict=False) as f:
        n_inlines = len(f.ilines)
        n_crosslines = len(f.xlines)
        
        survey_mask = np.zeros((n_inlines, n_crosslines), dtype=np.uint8)
        indices_to_process = range(0, n_inlines, STEP)
        total_steps = len(indices_to_process)
        start_time = time.time()

        for count, i in enumerate(indices_to_process):
            inline_idx = f.ilines[i]
            data = f.iline[inline_idx]

            # --- THE FIX ---
            # Round floats (1.0039) to nearest int (1) before checking
            data = np.round(data).astype(int)

            mask = np.isin(data, VALID_VALUES)
            has_data = np.any(mask, axis=1)

            survey_mask[i, :] = has_data.astype(np.uint8)

            if STEP > 1 and i + STEP < n_inlines:
                survey_mask[i:i+STEP, :] = has_data.astype(np.uint8)

            if count % 10 == 0 and count > 0:
                elapsed = time.time() - start_time
                avg_time_per_step = elapsed / count
                remaining_steps = total_steps - count
                secs_left = int(remaining_steps * avg_time_per_step)
                eta = str(datetime.timedelta(seconds=secs_left))
                print(f"Progress: {count}/{total_steps} | ETA: {eta}      ", end='\r')

        print(f"\nDone! Saved to {OUTPUT_MASK_PATH}")
    
    np.save(OUTPUT_MASK_PATH, survey_mask)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(survey_mask.T, cmap='gray', origin='lower', aspect='auto')
    plt.title(f"Survey Mask (Step={STEP})\nNow including Water (1) and Salt (4)")
    plt.xlabel("Inline")
    plt.ylabel("Crossline")
    plt.savefig(OUTPUT_IMAGE_PATH)
    print(f"Preview saved: {OUTPUT_IMAGE_PATH}")

if __name__ == "__main__":
    generate_mask_fast()