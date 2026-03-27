# Laser Triangulation Evaluation Tool

## Overview

This project implements a complete processing pipeline for laser triangulation measurements.

It is designed to take raw measurement images, detect laser points, refine their positions, assign indices, and reconstruct the corresponding 3D geometry using triangulation.

The pipeline is compatible with both simulated data and real measurement data.

---

## Features

* Automatic image loading and preprocessing
* Peak detection of laser spots
* Subpixel refinement using fitting methods
* Grid-based indexing of laser points
* 3D reconstruction via triangulation
* Validation against ground truth (simulation mode)
* Visualization tools for debugging and analysis

---

## Processing Pipeline

The evaluation follows these steps:

### 1. Input Handling

* PNG images are loaded and converted into `.npy` arrays
* Existing `.npy` files are reused for efficiency

---

### 2. Peak Detection

* Coarse detection of laser spots
* Configurable thresholds and neighborhood filters

---

### 3. Subpixel Fitting

* Refinement of detected peaks
* Methods:

  * Gaussian fit
  * Threshold centroid

Output:

* Laser Point Centers (LPC)

---

### 4. Index Assignment

* Points are mapped to a structured grid
* Center-based coordinate system:

```
(-2, -2) ... (0,0) ... (2,2)
```

* Works even with missing points
* Rotation and spacing are estimated automatically

---

### 5. Triangulation

Reconstruction of 3D points using:

* Laser ray direction (from metadata)
* Camera ray (from pixel coordinates)

The 3D point is computed as the closest point between:

* Laser ray
* Camera ray

---

### 6. Output

Generated files:

* Detected peaks (`*_detected_peaks.npy`)
* Fitted centers (`*_fitted_centers.npy`)
* Indexed points (`*_indexed_lpc.npy`)
* Triangulated 3D points (`*_triangulated_points.npy`)
* Visualization plots (`*.png`)

---

## Project Structure

```
laser_triangulation/
├─ main.py
├─ src/
│  ├─ detection/
│  ├─ fitting/
│  ├─ triangulation/
│  ├─ visualization/
│  ├─ io/
│  └─ utils/
├─ data/
│  ├─ input/
│  │  └─ images/
│  └─ output/
├─ README.md
└─ .gitignore
```

---

## Data Format

### Input

* PNG images or `.npy` arrays
* Stored in:

```
data/input/images/<timestamp>/
```

---

### Output

Each processed dataset produces:

#### Indexed LPC

```
[idx_x, idx_y, u, v]
```

#### Triangulated points

```
[idx_x, idx_y, x, y, z, u, v, line_distance]
```

---

## Usage

Run the pipeline:

```bash
python main.py
```

---

## Validation (Simulation Mode)

If ground truth data is available:

* Results are compared against:

```
ground_truth_points.npy
```

* Includes:

  * Index consistency check
  * Absolute and relative error analysis
  * Robust handling of near-zero values

---

## Visualization

The tool provides:

* Detection + fitting plots
* 3D reconstruction plots
* Color-coded depth visualization

---

## Notes

* Designed to work with simulated and real data
* Robust against missing points in the grid
* Assumes calibrated camera parameters from metadata

---

## Future Improvements

* Camera calibration integration
* Multi-camera support
* Real-time processing
* Improved noise robustness
* Automated parameter tuning

---

## Author

Developed as part of a laser triangulation project.
