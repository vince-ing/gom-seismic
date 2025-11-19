import os
import numpy as np
import segyio
import torch
from torch.utils.data import Dataset

# ================= CONFIGURATION =================
# Class mappings: Original Value -> Model Class
# 0 (Rock) -> 0
# 1 (Water) -> 1
# 4 (Salt)  -> 2
LABEL_MAPPING = {0: 0, 1: 1, 4: 2}
NUM_CLASSES = 3

# Normalization Constants (You might need to tune these based on your raw data)
# Typical seismic amplitude range approximation
SEISMIC_MIN = -2000.0
SEISMIC_MAX = 2000.0
# =================================================

class SaltDataset(Dataset):
    def __init__(self, raw_path, label_path, mask_path, patch_size=(256, 256), transform=None, limit_to_inlines=None):
        """
        Args:
            raw_path (str): Path to raw_seismic.segy
            label_path (str): Path to salt_mask.segy
            mask_path (str): Path to survey_mask.npy (created by make_mask.py)
            patch_size (tuple): Size of the 2D patch to crop (Depth, Crossline)
            limit_to_inlines (list): Optional list of inline indices to use (for Validation split)
        """
        self.raw_path = raw_path
        self.label_path = label_path
        self.patch_height, self.patch_width = patch_size
        self.transform = transform

        # 1. Load the Survey Mask
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Survey mask not found at {mask_path}. Run make_mask_v2.py first!")
        
        self.full_mask = np.load(mask_path) # Shape: (n_inlines, n_crosslines)
        
        # 2. Identify "Valid" Inlines (Inlines that contain data)
        # We sum the rows. If a row has mostly 0s, it's empty space.
        # We require at least 'patch_width' amount of valid data to bother looking at it.
        valid_mask_indices = np.where(self.full_mask.sum(axis=1) > self.patch_width)[0]
        
        # Filter if we are splitting Train/Val
        if limit_to_inlines is not None:
            self.valid_inlines = [i for i in valid_mask_indices if i in limit_to_inlines]
        else:
            self.valid_inlines = valid_mask_indices

        print(f"Dataset Initialized. Found {len(self.valid_inlines)} valid Inlines out of {self.full_mask.shape[0]}.")

        # We don't open SEGYs here to keep the object picklable for DataLoader workers.
        # We open them in __getitem__

    def __len__(self):
        # We define the "length" as the number of valid inlines.
        # In a real epoch, you might want to multiply this by X if you crop multiple patches per inline.
        return len(self.valid_inlines)

    def _open_files(self):
        """Helper to open files efficiently."""
        # strict=False is critical for slightly malformed exports
        raw_f = segyio.open(self.raw_path, ignore_geometry=False, strict=False)
        label_f = segyio.open(self.label_path, ignore_geometry=False, strict=False)
        return raw_f, label_f

    def __getitem__(self, idx):
        # 1. Pick an Inline Index
        inline_idx = self.valid_inlines[idx]
        
        # 2. Open Files (Lazy Loading)
        # Context manager ensures they close after reading, keeping memory clean
        with segyio.open(self.raw_path, strict=False) as raw_f, \
             segyio.open(self.label_path, strict=False) as label_f:
            
            # Get the actual segy line number (might not be 0,1,2... could be 1000, 1001...)
            actual_il = raw_f.ilines[inline_idx]

            # 3. Load the Full Slice (Samples x Crosslines)
            # Note: segyio returns (Crosslines, Samples), we usually want (Samples, Crosslines)
            raw_slice = raw_f.iline[actual_il].T
            label_slice = label_f.iline[actual_il].T

        # 4. Random Crop (Patching)
        # We need a patch of size (patch_height, patch_width)
        img_h, img_w = raw_slice.shape
        
        # If slice is smaller than patch, pad it (Edge case)
        if img_h < self.patch_height or img_w < self.patch_width:
            pad_h = max(0, self.patch_height - img_h)
            pad_w = max(0, self.patch_width - img_w)
            raw_slice = np.pad(raw_slice, ((0, pad_h), (0, pad_w)), mode='constant')
            label_slice = np.pad(label_slice, ((0, pad_h), (0, pad_w)), mode='constant')
            img_h, img_w = raw_slice.shape

        # Pick random top-left corner
        # Optimization: Try to pick a spot that actually has valid data (using the mask)
        # For now, simple random crop:
        start_h = np.random.randint(0, img_h - self.patch_height + 1)
        start_w = np.random.randint(0, img_w - self.patch_width + 1)

        raw_patch = raw_slice[start_h : start_h + self.patch_height, start_w : start_w + self.patch_width]
        label_patch = label_slice[start_h : start_h + self.patch_height, start_w : start_w + self.patch_width]

        # 5. Preprocessing & Cleaning
        
        # A. Fix the "1.0039" float issue by rounding to nearest integer
        label_patch = np.round(label_patch).astype(int)
        
        # B. Remap Labels (0,1,4 -> 0,1,2)
        # We use a fast numpy lookup
        # Initialize with 0 (Background)
        new_label = np.zeros_like(label_patch)
        for orig, target in LABEL_MAPPING.items():
            new_label[label_patch == orig] = target
            
        # C. Normalize Seismic Data (Min-Max scaling to 0-1)
        # Clip outliers first
        raw_patch = np.clip(raw_patch, SEISMIC_MIN, SEISMIC_MAX)
        # Scale to 0-1
        raw_patch = (raw_patch - SEISMIC_MIN) / (SEISMIC_MAX - SEISMIC_MIN)
        # Ensure float32
        raw_patch = raw_patch.astype(np.float32)

        # 6. Convert to PyTorch Tensors
        # Image needs (Channels, Height, Width). Seismic is 1 channel.
        image_tensor = torch.from_numpy(raw_patch).unsqueeze(0) # Add channel dim -> (1, H, W)
        label_tensor = torch.from_numpy(new_label).long()       # Classes -> (H, W)

        if self.transform:
            # Apply augmentations if any
            pass 

        return image_tensor, label_tensor

# ================= TEST BLOCK =================
if __name__ == "__main__":
    # Quick test to verify dimensions
    import matplotlib.pyplot as plt
    
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RAW_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'CroppedData.segy')
    LABEL_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'SaltMasksegy')
    MASK_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'survey_mask.npy')

    print("Testing Data Loader...")
    try:
        ds = SaltDataset(RAW_PATH, LABEL_PATH, MASK_PATH)
        img, lbl = ds[0] # Get first item
        
        print(f"Success! Returned Tensors:")
        print(f"Image Shape: {img.shape} (Should be 1, 256, 256)")
        print(f"Label Shape: {lbl.shape} (Should be 256, 256)")
        print(f"Label Unique Values: {torch.unique(lbl)}")
        
        # Show the patch
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Raw Seismic (Normalized)")
        plt.imshow(img.squeeze(), cmap='gray', aspect='auto')
        plt.subplot(1, 2, 2)
        plt.title("Label (Remapped 0,1,2)")
        plt.imshow(lbl, cmap='jet', aspect='auto')
        plt.show()
        
    except Exception as e:
        print(f"Test Failed: {e}")
        print("Ensure 'CroppedData.segy' exists in data/raw/!")