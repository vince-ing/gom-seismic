import os
import torch
import numpy as np
import segyio
from tqdm import tqdm
from data_loader import SaltDataset

# ================= CHUNKED CONFIGURATION =================
TOTAL_PATCHES = 5000     
PATCH_SIZE = (1024, 256) 
CHUNK_SIZE = 1000  # Save to disk every 1000 patches

QUOTAS = {
    'salt':  2000,
    'water': 2000,
    'random': 1000
}

LABEL_MAPPING = {0: 0, 1: 1, 4: 2}
SEISMIC_MIN = -2000.0
SEISMIC_MAX = 2000.0
# =========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'CroppedData.segy')
LABEL_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'SaltMask.segy')
MASK_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'survey_mask.npy')
CACHE_DIR = r"C:\Users\ig-gbds\Documents\SeismicProject\data\cache_chunks"

def create_cache_chunked():
    print(f"{'='*40}")
    print(f"📦 CHUNKED CACHE GENERATOR ({PATCH_SIZE})")
    print(f"Goal: {TOTAL_PATCHES} patches in chunks of {CHUNK_SIZE}")
    print(f"{'='*40}")

    if not os.path.exists(RAW_PATH):
        print(f"❌ Error: Raw file not found: {RAW_PATH}")
        return

    os.makedirs(CACHE_DIR, exist_ok=True)

    ds = SaltDataset(RAW_PATH, LABEL_PATH, MASK_PATH, patch_size=PATCH_SIZE)
    valid_indices = ds.valid_inlines
    
    collected_images = []
    collected_labels = []
    counts = {'salt': 0, 'water': 0, 'random': 0}
    total_collected = 0
    chunk_idx = 0
    
    print("Opening SEGY files...")
    
    try:
        with segyio.open(RAW_PATH, strict=False) as raw_f, \
             segyio.open(LABEL_PATH, strict=False) as label_f:
             
            pbar = tqdm(total=TOTAL_PATCHES, desc="Collecting")
            
            for dataset_idx in valid_indices:
                if total_collected >= TOTAL_PATCHES:
                    break
                
                # Quota Check
                salt_full = counts['salt'] >= QUOTAS['salt']
                water_full = counts['water'] >= QUOTAS['water']
                random_full = counts['random'] >= QUOTAS['random']
                
                if salt_full and water_full and random_full:
                    break

                actual_il = raw_f.ilines[dataset_idx]
                raw_slice = raw_f.iline[actual_il].T
                label_slice = label_f.iline[actual_il].T
                
                img_h, img_w = raw_slice.shape
                ph, pw = PATCH_SIZE
                
                if img_h < ph or img_w < pw: continue

                # Try 10 random crops per slice
                for _ in range(10):
                    start_h = np.random.randint(0, img_h - ph + 1)
                    start_w = np.random.randint(0, img_w - pw + 1)

                    label_patch = label_slice[start_h : start_h + ph, start_w : start_w + pw]
                    label_patch = np.round(label_patch).astype(int)
                    
                    new_label = np.zeros_like(label_patch)
                    for orig, target in LABEL_MAPPING.items():
                        new_label[label_patch == orig] = target
                        
                    has_salt = (2 in new_label)
                    has_water = (1 in new_label)
                    
                    category = None
                    if has_salt and counts['salt'] < QUOTAS['salt']:
                        category = 'salt'
                    elif has_water and counts['water'] < QUOTAS['water']:
                        category = 'water'
                    elif counts['random'] < QUOTAS['random']:
                        category = 'random'
                    
                    if category:
                        raw_patch = raw_slice[start_h : start_h + ph, start_w : start_w + pw]
                        raw_patch = np.clip(raw_patch, SEISMIC_MIN, SEISMIC_MAX)
                        raw_patch = (raw_patch - SEISMIC_MIN) / (SEISMIC_MAX - SEISMIC_MIN)
                        raw_patch = raw_patch.astype(np.float32)

                        collected_images.append(torch.from_numpy(raw_patch).unsqueeze(0))
                        collected_labels.append(torch.from_numpy(new_label).long())
                        
                        counts[category] += 1
                        total_collected += 1
                        pbar.update(1)
                        pbar.set_postfix(s=counts['salt'], w=counts['water'], r=counts['random'])
                        
                        # --- CHUNK SAVE TRIGGER ---
                        if len(collected_images) >= CHUNK_SIZE:
                            chunk_path = os.path.join(CACHE_DIR, f'chunk_{chunk_idx}.pt')
                            # Stack just this chunk
                            final_images = torch.stack(collected_images)
                            final_labels = torch.stack(collected_labels)
                            
                            torch.save({'images': final_images, 'labels': final_labels}, chunk_path)
                            
                            # Clear RAM
                            collected_images = []
                            collected_labels = []
                            chunk_idx += 1
                            
                        if total_collected >= TOTAL_PATCHES:
                            break

            pbar.close()

    except KeyboardInterrupt:
        print("\n⚠️ INTERRUPTED! Saving leftovers...")
    
    # Save leftovers
    if len(collected_images) > 0:
        chunk_path = os.path.join(CACHE_DIR, f'chunk_{chunk_idx}.pt')
        final_images = torch.stack(collected_images)
        final_labels = torch.stack(collected_labels)
        torch.save({'images': final_images, 'labels': final_labels}, chunk_path)
        print(f"Saved final chunk {chunk_idx}")

    print(f"\n✅ DONE! Saved chunks to {CACHE_DIR}")

if __name__ == "__main__":
    create_cache_chunked()