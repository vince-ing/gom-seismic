# Seismic Facies Classification with U-Net (Gulf of Mexico)

**Author:** Venancio Ingersoll  
**Date:** December 2025

> *Above: An animation of the model's predictions moving through the seismic volume. Top: Raw Seismic + Manual Ground Truth (Gold). Bottom: Model Probability Prediction (Jet Colormap).*

---

##  Project Overview

Identifying salt bodies in seismic reflection data is a critical but labor-intensive task in geophysical exploration. Salt structures in the Gulf of Mexico often have complex shapes (diapirs, canopies, overhangs) and steep dips that obscure subsalt imaging, making manual interpretation time-consuming and subjective.

This project implements a **U-Net Deep Learning model** to automate the pixel-level segmentation of salt bodies.

- **Goal:** To produce a probability map of salt presence in 2D seismic lines.
- **Region:** Mississippi Canyon, Gulf of Mexico.
- **Tech Stack:** PyTorch, Segyio, NumPy, Matplotlib.

---

##  Motivation

### Why do we care about salt?

1. **Hydrocarbon Traps:** Salt is impermeable and often forms structural traps where oil and gas accumulate. Accurately mapping salt flanks is essential for exploration.
2. **Geohazards:** Salt tectonics deform the seafloor, creating hazards for drilling and infrastructure.
3. **Velocity Modeling:** Salt has a much higher seismic velocity than surrounding sediments. Incorrect salt models lead to poor migration and blurry subsalt images.

---

##  Dataset & Preprocessing

The model was trained on public data from the **USGS National Archive of Marine Seismic Surveys (NAMSS)**.

- **Survey:** B-07H-00-LA (Gulf of Mexico)
- **Labels:** Ground truth masks were manually interpreted and tagged using Petrel.
- **Data Curation:**
  - The full 12GB+ volume was too large for direct training.
  - We generated **5,000 cropped samples** (1024x256 pixels).
  - **Balancing:** To prevent class imbalance (mostly background), crops were filtered to ensure at least 40% salt presence.
- **Preprocessing:** Robust normalization (clipping at 2.5 standard deviations) was applied to handle high-amplitude reflection outliers.

---

##  Methodology

Utilized a standard **U-Net architecture** implemented in PyTorch. The U-Net is ideal for segmentation tasks in geophysics because it preserves spatial resolution via skip connections, allowing the model to see both high-level context (salt body shape) and low-level texture (chaotic signal character).

### Key Libraries:

- **PyTorch:** Model training and inference (CUDA accelerated).
- **Segyio:** Reading and writing industrial SEG-Y seismic files.
- **Scikit-image & Scipy:** Image processing and resampling.
- **Pillow:** Generating result animations.

---

##  How to Run

This repository includes a sample dataset and a Jupyter Notebook to demonstrate the model's inference capabilities.

### 1. Clone the Repository

```bash
git clone https://github.com/vince-ing/gom-seismic.git
cd gom-seismic
```

### 2. Install Dependencies

It is recommended to use a virtual environment (Python 3.10+).

```bash
pip install -r requirements.txt
```

### 3. Run the Notebook

Launch the main presentation notebook:

```bash
jupyter notebook SaltML.ipynb
```

---

##  Results

The model showed significant improvements over traditional auto-tracking methods.

- **Texture Recognition:** The model successfully identified salt based on its chaotic internal texture, even in areas where the amplitude boundary was weak.
- **Discovery:** In several instances, the model identified potential deep salt structures that were missed during the initial manual tagging phase.

---

##  Future Work

- **Generalization:** Train on multiple surveys with different frequency contents to create a "universal" salt picker.
- **3D Consistency:** Extend the 2D U-Net to 3D (V-Net) or use recurrent layers (ConvLSTM) to enforce continuity across inline slices.
- **Mesh Generation:** Convert the output probability volumes into 3D meshes for direct import into modeling software.

---

##  Acknowledgments

Special thanks to the **Jackson School of Geosciences (GBDS)** at UT Austin for computational resources and guidance on seismic interpretation workflows.
