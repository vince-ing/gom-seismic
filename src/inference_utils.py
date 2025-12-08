import torch
import numpy as np
import segyio
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from src.model import UNet

# CONFIG DEFAULTS
SIGMA_CLIP_MODEL = 2.5 
VISUAL_SEISMIC_CLIP = 0.3 
VISUAL_GAMMA_PROB = 0.5 

def robust_normalize_tile(img):
    """Normalize data for the AI Model (Mean=0, Std=1)."""
    mean = np.mean(img)
    std = np.std(img)
    img = np.clip(img, mean - SIGMA_CLIP_MODEL * std, mean + SIGMA_CLIP_MODEL * std)
    min_val, max_val = np.min(img), np.max(img)
    if max_val - min_val == 0: return np.zeros_like(img)
    return 2 * (img - min_val) / (max_val - min_val) - 1.0

class SeismicHunter:
    def __init__(self, model_path, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"   🚀 Loading Model on {self.device}...")
        self.model = UNet(n_channels=1, n_classes=3).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def predict_slice(self, slice_data):
        """Runs inference on a single 2D seismic slice."""
        # 1. Normalize
        slice_norm = robust_normalize_tile(slice_data)

        # 2. Pad to 1024 height (Model Requirement)
        h, w = slice_norm.shape
        target_h = 1024
        
        if h < target_h:
            pad_h = np.zeros((target_h, w))
            pad_h[:h, :] = slice_norm
            input_data = pad_h
        else:
            input_data = slice_norm[:target_h, :]

        # Pad Width (Multiple of 32)
        target_w = ((w + 31) // 32) * 32
        pad_w = target_w - w
        if pad_w > 0:
            input_data = np.pad(input_data, ((0,0), (0, pad_w)), mode='constant')

        # 3. Predict
        tensor = torch.from_numpy(input_data).float().unsqueeze(0).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            out = self.model(tensor)
            prob = torch.softmax(out, dim=1)
            salt_prob = prob[0, 2, :, :].cpu().numpy()

        # 4. Cleanup
        salt_prob = salt_prob[:, :w]       
        if h < target_h: 
            salt_prob = salt_prob[:h, :]   
            
        return salt_prob

def load_segy_volume(filepath):
    """
    Loads a SEG-Y file into a 3D Numpy array (Inlines, Depth, Traces).
    Handles variable trace counts by cropping to the minimum common width.
    """
    if not filepath or filepath == "":
        return None
        
    print(f"📖 Reading {filepath}...")
    with segyio.open(filepath, "r", ignore_geometry=True) as src:
        # Get all inline numbers
        inlines = src.attributes(segyio.TraceField.INLINE_3D)[:]
        unique_ils = np.unique(inlines)
        n_inlines = len(unique_ils)
        
        # Determine dimensions by checking the first inline
        depth = len(src.samples)
        
        # Scan to find the minimum number of traces per inline (to handle jagged edges)
        min_traces = float('inf')
        for il in unique_ils:
            count = np.sum(inlines == il)
            if count < min_traces: min_traces = count
            
        print(f"   - Volume Size: {n_inlines} Inlines x {depth} Depth x {min_traces} Traces")
        
        volume = np.zeros((n_inlines, depth, min_traces), dtype=np.float32)
        
        for i, il in enumerate(unique_ils):
            # Find indices for this inline
            indices = np.where(inlines == il)[0]
            
            # Read traces
            # Note: We only read up to 'min_traces' to keep the cube rectangular
            for j in range(min_traces):
                volume[i, :, j] = src.trace[indices[j]]
                
    return volume

def create_prediction_plot(seismic, prob, mask=None, figsize=(12, 10)):
    """Generates a matplotlib figure for the notebook."""
    fig, axs = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # Top: Seismic (+ Ground Truth if provided)
    axs[0].imshow(seismic, cmap='gray', aspect='auto', 
                  vmin=-VISUAL_SEISMIC_CLIP, vmax=VISUAL_SEISMIC_CLIP)
    
    if mask is not None:
        mask_overlay = np.ma.masked_where(mask < 0.5, mask)
        axs[0].imshow(mask_overlay, cmap='viridis', aspect='auto', alpha=0.6)
        axs[0].set_title("Seismic Input + Ground Truth (Gold)", fontweight='bold')
    else:
        axs[0].set_title("Seismic Input", fontweight='bold')
        
    axs[0].set_ylabel("Time/Depth")

    # Bottom: Prediction Overlay
    axs[1].imshow(seismic, cmap='gray', aspect='auto', 
                  vmin=-VISUAL_SEISMIC_CLIP, vmax=VISUAL_SEISMIC_CLIP)
    
    # Gamma Correct Probability for visibility
    prob_boosted = np.power(prob, VISUAL_GAMMA_PROB)
    prob_masked = np.ma.masked_where(prob_boosted < 0.05, prob_boosted)
    
    axs[1].imshow(prob_masked, cmap='jet', aspect='auto', alpha=0.5, vmin=0.0, vmax=1.0)
    axs[1].set_title("Model Prediction (Jet)", fontweight='bold')
    axs[1].set_ylabel("Time/Depth")
    
    plt.tight_layout()
    return fig