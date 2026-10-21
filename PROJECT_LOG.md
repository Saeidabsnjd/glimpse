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
* **Core Topic**: Real-Time Generative Neural Rendering and Single-Step Distillation for Mixed Reality
* **Primary Stack**: Python 3.10+, Blender 4.x/5.x (`bpy`), PyTorch (CUDA), NumPy, OpenEXR

---

## 2. Codebase Architecture & State

```
glimpse/
├── .gitignore                      Ignore rules (ignores heavy PDFs, cache, blend1, renders)
├── README.md                       Project overview and environment setup
├── Mathematical_Foundations.md      Living mathematical derivations handbook (BSSRDF, Look-At, Pinhole)
├── PROJECT_LOG.md                  This master engineering & continuity log
├── assets/
│   └── camera_hemisphere_orbit.jpg  Scientific 3D diagram of camera orbit dome
└── src/
    ├── __init__.py                 Root package
    ├── data/
    │   ├── __init__.py             Data generation and loader package
    │   └── camera_orbit.py         [COMPLETED] Spherical-to-Cartesian, Look-At matrix, hemisphere orbit
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
* **Status**: Fully implemented and mathematically verified.
* **Key Functions**:
  * `spherical_to_cartesian(radius, azimuth_deg, elevation_deg)`:
    Converts spherical angles to $(x, y, z)$ on the upper dome using horizontal slice radius $R \cdot \cos(\phi)$ and vertical height $R \cdot \sin(\phi)$.
  * `compute_look_at_matrix(camera_pos, target_pos, world_up)`:
    Computes unit forward vector $\mathbf{f}$, horizontal right vector $\mathbf{r} = \mathbf{f} \times \mathbf{u}_{\text{world}}$, and orthogonal up vector $\mathbf{u} = \mathbf{r} \times \mathbf{f}$. Handles singular zenith edge case.
  * `camera_pose_to_extrinsic_4x4(position, rotation)`:
    Assembles standard $4 \times 4$ transform matrix $[R \mid \mathbf{t}]$ for NeRF/transforms.json compatibility.
  * `generate_hemisphere_orbit(radius, n_azimuth, n_elevation, min_el, max_el)`:
    Generates multi-view orbits (e.g., 48 viewpoints across 12 azimuths $\times$ 4 elevations).

---

## 5. Next Immediate Milestone: The Blender Synthetic Data Engine

* **Target File**: `src/data/dataset_engine.py` (or `blender_render_pipeline.py`)
* **Objective**: Automate Blender via `bpy` to render paired datasets:
  1. Load 3D head/character asset with calibrated Subsurface Scattering (SSS) material at $(0, 0, 0)$.
  2. Instantiate virtual camera and loop through viewpoints generated by `camera_orbit.py`.
  3. Configure Blender Compositor node tree:
     * **Target Ground Truth**: Cycles path-traced at 512 samples per pixel.
     * **Input Draft**: Cycles at 2 samples per pixel (or fast EEVEE rasterizer).
     * **Conditioning Pass 1**: Metric Z-Depth pass.
     * **Conditioning Pass 2**: Camera-space surface normals.
  4. Save outputs to disk in multi-layer OpenEXR / 16-bit PNG format alongside `transforms.json`.

---

## 6. Onboarding Protocol for Antigravity on Lab PC

When starting work on the TU Dresden lab workstation:
1. Clone the repository:
   ```bash
   git clone https://github.com/Saeidabsnjd/glimpse.git
   cd glimpse
   ```
2. Feed this file to Antigravity with the prompt:
   > *"I am working on the glimpse project. Read `PROJECT_LOG.md` and `Mathematical_Foundations.md`. We just completed `src/data/camera_orbit.py`. Now let's continue with Step 5: building `src/data/dataset_engine.py` in Blender."*
