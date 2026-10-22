# glimpse: Engineering Log & Lab Continuity Protocol

This document serves as the master engineering log, architectural state record, and agent continuity handbook for the **glimpse** research project.

When switching to the **IXLAB workstation PC at TU Dresden**, provide this file to Antigravity so it immediately inherits the complete project state, conventions, and next steps without needing prior conversation context.

---

## 1. Project Metadata
* **Project Name**: `glimpse`
* **Repository**: [https://github.com/Saeidabsnjd/glimpse.git](https://github.com/Saeidabsnjd/glimpse.git)
* **Candidate**: Saeid Abbasnejad (ICT Engineering, University of Padova)
* **Host Institution**: Immersive Experience Lab (IXLAB), Faculty of Computer Science, TU Dresden
* **Supervisors**: Prof. Dr. Matthew McGinity, Florian Schier, M.Sc.
* **Core Research Objective**: Real-Time Generative Neural Rendering and Single-Step Distillation for Mixed Reality (targeting sub-15ms inference, >60 FPS, global illumination, contact shadows, and complex optical materials including SSS)
* **Primary Stack**: Python 3.10+, Blender 4.x/5.x (`bpy`), PyTorch (CUDA), NumPy, OpenEXR

---

## 2. Codebase Architecture & State

```
glimpse/
├── .gitignore                      Ignore rules (ignores heavy PDFs, cache, blend1, renders)
├── README.md                       Project overview and environment setup
├── Mathematical_Foundations.md      Living mathematical derivations handbook (BSSRDF, Look-At, Pinhole)
├── PIPELINE_OVERVIEW.md            Comprehensive end-to-end system pipeline & architecture specification
├── PROJECT_LOG.md                  This master engineering & continuity log
├── assets/
│   └── camera_hemisphere_orbit.jpg  Scientific 3D diagram of camera orbit dome
└── src/
    ├── __init__.py                 Root package
    ├── data/
    │   ├── __init__.py             Data generation and loader package
    │   ├── camera_orbit.py         [COMPLETED] Spherical-to-Cartesian, Look-At matrix, hemisphere orbit
    │   └── dataset_engine.py       [COMPLETED] Headless Blender 5.x engine with OptiX GPU, SSS material, compositor wiring
    ├── models/
    │   └── __init__.py             [PENDING] Single-step generative neural rendering backbones
    ├── losses/
    │   └── __init__.py             [PENDING] Photometric, perceptual (LPIPS), and SSS material losses
    └── utils/
        └── __init__.py             [PENDING] Camera projection, EXR I/O, disparity transforms
```

---

## 3. Engineering & Mathematical Conventions Established

All future modules written for this codebase must strictly adhere to these verified conventions:

1. **Camera Coordinate System (OpenGL / Blender standard)**:
   * **+X**: Points to the camera's Right.
   * **+Y**: Points to the camera's Up.
   * **-Z**: Points along the optical axis into the scene (view direction towards target).
   * **Extrinsic Matrix $[R \mid \mathbf{t}]$**: Sized $4 \times 4$ in `float32`. Columns of $R$ are $[\mathbf{r}, \mathbf{u}, -\mathbf{f}]$.

2. **Numerical Precision**:
   * All geometry calculations and tensor inputs are standardized to **`float32`** (`np.float32` / `torch.float32`) for GPU and Tensor Core efficiency.

3. **G-Buffer Representation (View-Invariance)**:
   * **Surface Normals**: Expressed in **Camera (View) Space**, not World Space, ensuring consistent shading regardless of camera orbit angle.
   * **Depth Buffer**: Metric depth $Z$ in meters converted to **disparity** $d = \frac{1}{Z}$ to bound values to $[0, 1]$ and prevent exploding gradients.

4. **Coordinate Frame Orthogonality**:
   * Any camera rotation matrix $R$ must satisfy:
     $$R \cdot R^T = I \quad \text{and} \quad \det(R) = +1.0$$

---

## 4. Chronological Engineering Log

### Milestone 1: Repository Initialization & Mathematics Handbook
* **Action**: Configured clean Git repository for `glimpse` and linked to `https://github.com/Saeidabsnjd/glimpse.git`.
* **Action**: Created [Mathematical_Foundations.md](file:///c:/Users/Saeed/.gemini/antigravity/scratch/blender/Mathematical_Foundations.md) containing:
  * The Kajiya Rendering Equation & Monte Carlo error convergence law ($\sigma \propto \frac{1}{\sqrt{N}}$).
  * Subsurface Scattering (BSSRDF) vs. local BRDF reflection equations.
  * Extrinsic $4 \times 4$ and Intrinsic $3 \times 3$ camera projection matrices.
  * G-buffer normalization and metric-to-disparity transformations.
  * Embedded visual diagram of the upper hemisphere camera orbit dome.

### Milestone 2: Camera Orbit Engine (`src/data/camera_orbit.py`)
* **File**: [camera_orbit.py](file:///c:/Users/Saeed/.gemini/antigravity/scratch/blender/src/data/camera_orbit.py)
* **Status**: Fully implemented, tested, and pushed to GitHub.
* **Key Functions**:
  * `spherical_to_cartesian(radius, azimuth_deg, elevation_deg)`
  * `compute_look_at_matrix(camera_pos, target_pos, world_up)`
  * `camera_pose_to_extrinsic_4x4(position, rotation)`
  * `generate_hemisphere_orbit(radius, n_azimuth, n_elevation, min_el, max_el)`

### Milestone 3: End-to-End System Pipeline Specification (`PIPELINE_OVERVIEW.md`)
* **File**: [PIPELINE_OVERVIEW.md](file:///c:/Users/Saeed/.gemini/antigravity/scratch/blender/PIPELINE_OVERVIEW.md)
* **Status**: Complete architecture specification document.
* **Coverage**:
  * Stage 1: Blender Ground-Truth Synthetic Data Engine (Cycles 512 spp vs 2 spp, SSS materials, G-buffers).
  * Stage 2: Data Preprocessing & Tensor Layouts ($B \times 7 \times H \times W$, disparity $d = 1/Z$).
  * Stage 3: Two-Stream Generative Architecture (Appearance stream, Geometry stream, SPADE / Cross-Attention fusion).
  * Stage 4: Single-Step Generative Distillation (Rectified Flow / Consistency formulation for $<15\text{ ms}$ inference).
  * Stage 5: Multi-Objective Loss Formulations (Charbonnier, LPIPS, Sobel edge, physical SSS translucency).
  * Stage 6: Mixed Reality Deployment & Optimization (ONNX/TensorRT FP16 compilation for Meta Quest 3 / Vision Pro at $60+\text{ FPS}$).

### Milestone 4: Automated Blender Dataset Engine (`src/data/dataset_engine.py`)
* **File**: [dataset_engine.py](file:///c:/Users/Saeed/.gemini/antigravity/scratch/blender/src/data/dataset_engine.py)
* **Status**: Complete headless pipeline for Blender 5.x.
* **Key Mechanisms**:
  * Hardware Acceleration: Explicit OptiX GPU detection targeting NVIDIA RTX workstations with CUDA fallback for Turing GTX cards.
  * Material Calibration: Principled BSDF with calibrated human skin Subsurface Scattering (Radius: [1.0, 0.2, 0.1], Scale: 0.005m / 5mm, IOR: 1.4).
  * Multi-Pass Compositor: Uses dedicated dual output nodes for depth (`depth/view_XXX.exr`) and normal (`normal/view_XXX.exr`), guaranteeing unpartitioned single-part OpenEXR streams that load in sub-millisecond Python times.
  * Automated Orbit Rendering: Loops over viewpoints from `camera_orbit.py`, renders paired 512-spp ground truth and 2-spp draft inputs, and exports camera intrinsics/extrinsics to `transforms.json`.
  * **Critical Calibration (Denoising Toggle)**: Blender 5.2 enables OpenImageDenoise (`use_denoising = True`) by default. The engine now explicitly enforces `scene.cycles.use_denoising = False` on the draft pass to guarantee true raw, noisy Monte Carlo inputs while keeping `use_denoising = True` on the ground-truth reference.

---

## 5. Next Immediate Milestone: PyTorch Dataset Loader & Tensor Pipeline

* **Target File**: `src/data/dataset.py`
* **Objective**: Load and preprocess rendered datasets for model training:
  1. OpenEXR parsing for metric depth and camera-space normals via `OpenEXR` and `Imath`.
  2. Inverse disparity transformation: $d = 1 / Z_{\text{cam}}$ mapped to $[0, 1]$.
  3. Packaging into $B \times 7 \times H \times W$ conditioning tensors ready for GPU training.

---

## 6. Onboarding Protocol for Antigravity on Lab PC (RTX 3060 Ti)

When working on the TU Dresden lab workstation:
1. Pull the latest repository:
   ```bash
   git pull origin main
   ```
2. Launch the full high-sample dataset generation on the RTX 3060 Ti:
   ```cmd
   "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" --background --python src\data\dataset_engine.py -- --output_dir data\orbit48_gt512 --n_azimuth 12 --n_elevation 4 --samples_gt 512 --samples_draft 2 --resolution 512
   ```
