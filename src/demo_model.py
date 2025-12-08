import os
import torch
import numpy as np
import segyio
import matplotlib.pyplot as plt
from skimage import morphology
from skimage.morphology import disk
from model import UNet

# ================= USER CONFIGURATION =================
INPUT_FOLDER = r"G:\Working\Students\Undergraduate\For_Vince\SeismicFacies\namss.B-26-99-TX.mcs3d.airgun\Data\segy"
OUTPUT_FOLDER = r"G:\Working\Students\Undergraduate\For_Vince\SeismicFacies\Reports"

OUTPUT_FILENAME = "Salt_Discovery_Report.png"
OUTPUT_IMAGE = os.path.join(OUTPUT_FOLDER, OUTPUT_FILENAME)

MODEL_PATH = "models/best_model.pth"

# How many "Good" examples to find per file?
EXAMPLES_PER_FILE = 2 

# Threshold: Only show slice if salt covers > 1% of the image
# This prevents showing empty black screens.
MIN_SALT_COVERAGE = 0.01 
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

class SeismicHunter:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"   🚀 Loading Model...")
        self.model = UNet(n_channels=1, n_classes=3).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def predict_slice(self, slice_data):
        # Slice is (Height, Width)
        # Normalize
        slice_norm = robust_normalize_tile(slice_data)
        
        # To Tensor (1, 1, H, W)
        tensor = torch.from_numpy(slice_norm).float().unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            out = self.model(tensor)
            prob = torch.softmax(out, dim=1)
            salt_prob = prob[0, 2, :, :].cpu().numpy()
            
        # Cleanup
        binary = salt_prob > 0.5
        clean = morphology.remove_small_objects(binary, min_size=64)
        clean = morphology.binary_closing(clean, disk(3))
        
        return clean.astype(float), slice_norm

def scan_file(filepath, hunter, num_examples):
    filename = os.path.basename(filepath)
    print(f"\n📂 Scanning: {filename}")
    
    hits = [] 
    
    try:
        with segyio.open(filepath, "r", ignore_geometry=True) as src:
            src.mmap()
            
            # FAST METHOD: Don't map the whole file. Just check total traces.
            n_traces = src.tracecount
            print(f"   (File has {n_traces} traces. Jumping to random locations...)")
            
            # Try 50 random jumps
            for _ in range(50):
                if len(hits) >= num_examples:
                    break
                
                # Jump to a random trace index
                start_idx = np.random.randint(0, n_traces - 5000)
                
                # Grab a chunk of 2000 traces (enough for a few slices)
                # This is INSTANT because we don't scan headers first
                headers = src.header[start_idx : start_idx+2000]
                
                # Extract Inline numbers from this small chunk only
                chunk_inlines = [h[segyio.TraceField.INLINE_3D] for h in headers]
                unique_in_chunk = np.unique(chunk_inlines)
                
                # Process the first full inline found in this chunk
                for il in unique_in_chunk:
                    # Find traces in this chunk that match the inline
                    # Note: These are relative indices within our 2000-trace chunk
                    mask = (np.array(chunk_inlines) == il)
                    if np.sum(mask) < 100: # Skip fragments (need at least 100 traces for a good image)
                        continue
                        
                    # Calculate absolute indices in the file
                    rel_indices = np.where(mask)[0]
                    abs_indices = start_idx + rel_indices
                    
                    # Read Traces
                    slice_data = src.trace[abs_indices].T
                    
                    # Crop/Pad to 1024 depth
                    if slice_data.shape[0] > 1024:
                        slice_data = slice_data[:1024, :]
                    elif slice_data.shape[0] < 1024:
                        pad = np.zeros((1024, slice_data.shape[1]))
                        pad[:slice_data.shape[0], :] = slice_data
                        slice_data = pad

                    # Predict
                    mask_pred, seismic_norm = hunter.predict_slice(slice_data)
                    
                    # Check Salt Coverage
                    salt_ratio = np.sum(mask_pred) / mask_pred.size
                    
                    if salt_ratio > MIN_SALT_COVERAGE:
                        print(f"   ✅ FOUND SALT! Inline {il} (Coverage: {salt_ratio:.1%})")
                        hits.append((il, seismic_norm, mask_pred))
                        break # Found one in this chunk, move to next random jump
                    
    except Exception as e:
        print(f"   ⚠️ Reader Error: {e}")
        return []

    return hits

def main():
    if not os.path.exists(INPUT_FOLDER):
        print("❌ Input folder not found.")
        return

    hunter = SeismicHunter(MODEL_PATH)
    
    files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith('.segy') or f.endswith('.sgy')]
    print(f"Found {len(files)} files.")
    
    # Collect results
    all_results = {} # Filename -> List of hits
    
    for f in files:
        full_path = os.path.join(INPUT_FOLDER, f)
        hits = scan_file(full_path, hunter, EXAMPLES_PER_FILE)
        if hits:
            all_results[f] = hits
            
    # === GENERATE REPORT ===
    if not all_results:
        print("❌ No significant salt found in any file sampled.")
        return

    print("\n🎨 Generating Report...")
    total_rows = sum(len(hits) for hits in all_results.values())
    
    # Plot: Each Row = One Hit. Left = Seismic, Right = Prediction Overlay
    fig, axs = plt.subplots(total_rows, 2, figsize=(12, 4 * total_rows))
    plt.suptitle(f"Automated Salt Detection Report\n(Model: {os.path.basename(MODEL_PATH)})", fontsize=16)
    
    current_row = 0
    
    for filename, hits in all_results.items():
        for (il, seismic, mask) in hits:
            
            # 1. Seismic View
            ax_seis = axs[current_row, 0]
            ax_seis.imshow(seismic, cmap='gray', aspect='auto')
            ax_seis.set_title(f"{filename}\nInline {il}", fontsize=10, fontweight='bold')
            ax_seis.axis('off')
            
            # 2. Prediction View (Overlay)
            ax_pred = axs[current_row, 1]
            ax_pred.imshow(seismic, cmap='gray', aspect='auto')
            # Overlay yellow with transparency
            # Create RGBA
            overlay = np.zeros((mask.shape[0], mask.shape[1], 4))
            overlay[mask == 1] = [1, 1, 0, 0.5] # Yellow, 50% opacity
            
            ax_pred.imshow(overlay, aspect='auto')
            ax_pred.set_title("Predicted Salt (Yellow)", fontsize=10, fontweight='bold')
            ax_pred.axis('off')
            
            current_row += 1
            
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(OUTPUT_IMAGE, dpi=150)
    print(f"✅ Report Saved: {OUTPUT_IMAGE}")

if __name__ == "__main__":
    main()
