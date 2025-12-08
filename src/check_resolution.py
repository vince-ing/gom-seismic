import os
import segyio
import numpy as np

# ================= CONFIGURATION =================
# 1. Your Training File (The "Gold Standard")
TRAIN_FILE = r"G:\Working\Students\Undergraduate\For_Vince\SeismicFacies\gom-seismic\data\raw\CroppedData.segy"

# 2. The Folder with the 7 New Files
NEW_FILES_FOLDER = r"G:\Working\Students\Undergraduate\For_Vince\SeismicFacies\namss.B-26-99-TX.mcs3d.airgun\Data\segy"
# =================================================

def get_file_stats(filepath):
    try:
        with segyio.open(filepath, "r", ignore_geometry=True) as src:
            # 1. Get Sample Rate (Vertical Resolution)
            # Usually at byte 3217. Value is in microseconds (e.g., 4000 = 4ms)
            sample_rate = src.bin[segyio.BinField.Interval]
            
            # 2. Get Trace Length
            n_samples = src.samples.size
            
            return sample_rate, n_samples
    except Exception as e:
        return "Error", str(e)

def main():
    print(f"{'='*60}")
    print(f"📏 SEISMIC RESOLUTION CHECKER")
    print(f"{'='*60}")

    # 1. Check Training Data
    print(f"\n--- REFERENCE (TRAINING DATA) ---")
    if os.path.exists(TRAIN_FILE):
        rate, length = get_file_stats(TRAIN_FILE)
        print(f"File: {os.path.basename(TRAIN_FILE)}")
        print(f"   Vertical Sample Rate: {rate} microseconds ({rate/1000} ms)")
        print(f"   Trace Depth:          {length} pixels")
        ref_rate = rate
    else:
        print("⚠️ Training file not found. Cannot compare.")
        ref_rate = None

    # 2. Check New Files
    print(f"\n--- NEW FILES (GOM-SEISMIC) ---")
    files = [f for f in os.listdir(NEW_FILES_FOLDER) if f.endswith('.segy') or f.endswith('.sgy')]
    
    for f in files:
        full_path = os.path.join(NEW_FILES_FOLDER, f)
        rate, length = get_file_stats(full_path)
        
        status = "✅ MATCH"
        if ref_rate and rate != ref_rate:
            status = "❌ MISMATCH (TEXTURE WILL FAIL)"
        
        print(f"File: {f}")
        print(f"   Sample Rate: {rate} us | Depth: {length} px  -> {status}")

if __name__ == "__main__":
    main()
