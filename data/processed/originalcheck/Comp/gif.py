import os
from PIL import Image

# ================= USER CONFIGURATION =================
OUTPUT_NAME = "animation.gif"

# Speed of the animation in milliseconds per frame
# 100 ms = 10 frames per second (Fast/Smooth)
# 200 ms = 5 frames per second (Slower)
# 500 ms = 2 frames per second (Slideshow speed)
FRAME_DURATION = 800 
# ======================================================

def create_gif_from_folder():
    folder_path = os.getcwd()
    
    # Find all PNG files
    images = [img for img in os.listdir(folder_path) if img.lower().endswith(".png")]
    
    # Sort by length first, then name (handles frame_1, frame_10, frame_2 correctly)
    images.sort(key=lambda x: (len(x), x))
    
    if not images:
        print("❌ No PNG images found in this folder!")
        return

    print(f"🎥 Found {len(images)} frames.")
    print(f"⏱️  Creating GIF at {FRAME_DURATION}ms per frame...")
    
    # Open images
    frames = [Image.open(os.path.join(folder_path, img)) for img in images]
    
    # Save GIF
    output_path = os.path.join(folder_path, OUTPUT_NAME)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        optimize=True,
        duration=FRAME_DURATION, 
        loop=0
    )
    
    print(f"✅ GIF saved: {output_path}")

if __name__ == "__main__":
    create_gif_from_folder()
