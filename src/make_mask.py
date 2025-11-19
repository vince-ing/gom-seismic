import segyio
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import datetime

# ================= CONFIGURATION =================
LABEL_FILENAME = "SaltMask.segy" 

# STEP = 10 means "Read every 10th line". 
# Runs 10x faster. Great for testing the shape.
# Change to STEP = 1 for the final high-precision run.
STEP = 10 

# Values to consider "Valid Data" (Water=1, Salt=4)
VALID_VALUES = [1, 4]
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
    print(f"Starting Mask Generation (Step Size: {STEP})")
    print(f"{'='*40}")

    with segyio.open(INPUT_PATH, ignore_geometry=False, strict=False) as f:
        n_inlines = len(f.ilines)
        n_crosslines = len(f.xlines)
        
        # Initialize full-size mask with zeros
        survey_mask = np.zeros((n_inlines, n_crosslines), dtype=np.uint8)

        print(f"Volume: {n_inlines} x {n_crosslines} x {f.samples.size}")
        print("Scanning...")

        start_time = time.time()
        
        # We step through the file (e.g., 0, 10, 20, 30...)
        indices_to_process = range(0, n_inlines, STEP)
        total_steps = len(indices_to_process)

        for count, i in enumerate(indices_to_process):
            # Read Inline
            inline_idx = f.ilines[i]
            data = f.iline[inline_idx]

            # Check for ANY valid value (1 or 4) in the trace
            # isin is faster than multiple (data==1) | (data==4) checks
            mask = np.isin(data, VALID_VALUES)
            has_data = np.any(mask, axis=1)

            # Fill the current line in the mask
            survey_mask[i, :] = has_data.astype(np.uint8)

            # If we are skipping lines (Draft Mode), fill the gaps forward
            # This makes the image look solid instead of striped
            if STEP > 1 and i + STEP < n_inlines:
                survey_mask[i:i+STEP, :] = has_data.astype(np.uint8)

            # --- ETA CALCULATION ---
            if count % 10 == 0 and count > 0:
                elapsed = time.time() - start_time
                avg_time_per_step = elapsed / count
                remaining_steps = total_steps - count
                secs_left = int(remaining_steps * avg_time_per_step)
                eta = str(datetime.timedelta(seconds=secs_left))
                
                # Dynamic print on the same line
                print(f"Progress: {count}/{total_steps} | ETA: {eta}      ", end='\r')

        total_time = time.time() - start_time
        print(f"\n\nDone! Total time: {total_time:.1f} seconds")

    # Save
    print(f"Saving mask to {OUTPUT_MASK_PATH}...")
    np.save(OUTPUT_MASK_PATH, survey_mask)
    
    # Visualize
    plt.figure(figsize=(10, 10))
    plt.imshow(survey_mask.T, cmap='gray', origin='lower', aspect='auto')
    plt.title(f"Survey Mask (Step={STEP})\nWhite=Data, Black=Void")
    plt.xlabel("Inline")
    plt.ylabel("Crossline")
    plt.colorbar(label="Valid Data")
    plt.savefig(OUTPUT_IMAGE_PATH)
    print(f"Preview saved to {OUTPUT_IMAGE_PATH}")
    print("Check the image to ensure your polygon shape is correct!")

if __name__ == "__main__":
    generate_mask_fast()