"""
glimpse: Camera Orbit and Viewpoint Generator
Computes 3D camera positions and orthonormal Look-At rotation matrices
over an upper hemisphere for multi-view synthetic dataset rendering.
"""

import math
from typing import List, Dict
import numpy as np


def spherical_to_cartesian(
    radius: float, 
    azimuth_deg: float, 
    elevation_deg: float
) -> np.ndarray:
    """
    Converts spherical coordinates (radius, azimuth, elevation) to 3D Cartesian coordinates (x, y, z).

    Args:
        radius: Radial distance from origin to camera in meters.
        azimuth_deg: Horizontal rotation angle (0° to 360°).
        elevation_deg: Vertical angle from ground plane up to zenith (0° to 90°).

    Returns:
        np.ndarray of shape (3,) containing [x, y, z] in world coordinates.
    """
    azimuth_rad = math.radians(azimuth_deg)
    elevation_rad = math.radians(elevation_deg)

    x = radius * math.cos(elevation_rad) * math.sin(azimuth_rad)
    y = radius * math.cos(elevation_rad) * math.cos(azimuth_rad)
    z = radius * math.sin(elevation_rad)

    return np.array([x, y, z], dtype=np.float32)


def compute_look_at_matrix(
    camera_pos: np.ndarray, 
    target_pos: np.ndarray = np.array([0.0, 0.0, 0.0], dtype=np.float32),
    world_up: np.ndarray = np.array([0.0, 0.0, 1.0], dtype=np.float32)
) -> np.ndarray:
    """
    Computes a 3x3 rotation matrix R that orients the camera from camera_pos towards target_pos.
    
    Standard Computer Graphics Camera Convention (OpenGL / Blender):
        +X: Points to camera's right
        +Y: Points to camera's up
        -Z: Points along the optical axis into the scene (towards target)

    Args:
        camera_pos: 3D position of camera [x, y, z].
        target_pos: 3D point the camera is looking at (default: scene origin [0, 0, 0]).
        world_up: Global vertical reference vector (default: +Z [0, 0, 1]).

    Returns:
        R: 3x3 rotation matrix where columns are [right, up, -forward].
    """
    # 1. Forward vector: Points from camera towards the target
    forward = target_pos - camera_pos
    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-8:
        raise ValueError("Camera position and target position cannot be identical.")
    forward = forward / forward_norm

    # 2. Right vector: Perpendicular to both forward direction and world up
    right = np.cross(forward, world_up)
    right_norm = np.linalg.norm(right)
    
    # Handle the singular edge case: camera looking straight down from zenith (+Z)
    if right_norm < 1e-6:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        right = right / right_norm

    # 3. True Up vector: Perpendicular to both right and forward
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)

    # 4. Assemble 3x3 rotation matrix
    # Columns correspond to camera local axes expressed in world coordinates:
    # Column 0: +X (Right)
    # Column 1: +Y (Up)
    # Column 2: -Z (View direction)
    R = np.stack([right, up, -forward], axis=1)

    return R


def camera_pose_to_extrinsic_4x4(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """
    Assembles a 4x4 Camera-to-World extrinsic matrix [R | t] commonly used in NeRF/transforms.json.
    """
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = rotation
    T[:3, 3] = position
    return T


def generate_hemisphere_orbit(
    radius: float = 1.5,
    n_azimuth: int = 12,
    n_elevation: int = 4,
    min_elevation_deg: float = 5.0,
    max_elevation_deg: float = 65.0
) -> List[Dict[str, np.ndarray]]:
    """
    Generates a uniform grid of camera viewpoints over the upper hemisphere.

    Args:
        radius: Distance from object in meters.
        n_azimuth: Number of horizontal angular steps (0° to 360°).
        n_elevation: Number of vertical angular steps.
        min_elevation_deg: Lowest elevation angle (near table level).
        max_elevation_deg: Highest elevation angle (near top of head).

    Returns:
        List of dicts, each containing:
            - 'position': np.ndarray of shape (3,)
            - 'rotation': np.ndarray of shape (3, 3)
            - 'transform_matrix': np.ndarray of shape (4, 4)
            - 'azimuth_deg': float
            - 'elevation_deg': float
    """
    viewpoints = []
    
    azimuths = np.linspace(0.0, 360.0, n_azimuth, endpoint=False)
    elevations = np.linspace(min_elevation_deg, max_elevation_deg, n_elevation)

    for elevation in elevations:
        for azimuth in azimuths:
            pos = spherical_to_cartesian(radius, float(azimuth), float(elevation))
            rot = compute_look_at_matrix(pos)
            mat4 = camera_pose_to_extrinsic_4x4(pos, rot)

            viewpoints.append({
                "position": pos,
                "rotation": rot,
                "transform_matrix": mat4,
                "azimuth_deg": float(azimuth),
                "elevation_deg": float(elevation)
            })

    return viewpoints


if __name__ == "__main__":
    print("=" * 65)
    print("Testing glimpse Camera Orbit Generator")
    print("=" * 65)

    test_orbit = generate_hemisphere_orbit(radius=1.5, n_azimuth=4, n_elevation=3)
    print(f"Total viewpoints generated: {len(test_orbit)}\n")

    for i, vp in enumerate(test_orbit):
        pos = vp["position"]
        az = vp["azimuth_deg"]
        el = vp["elevation_deg"]
        print(f"View {i+1:02d} | Azimuth: {az:5.1f}° | Elevation: {el:5.1f}° | Pos: [{pos[0]:6.2f}, {pos[1]:6.2f}, {pos[2]:6.2f}]")

    print("\nMathematical Verification on View 1:")
    R0 = test_orbit[0]["rotation"]
    print("Orthonormality check (R @ R.T should be Identity matrix):")
    print(np.round(R0 @ R0.T, 3))

    det_R = np.linalg.det(R0)
    print(f"\nDeterminant of R (must be exactly +1.0 for rigid rotation): {det_R:.4f}")

    print("\nSample 4x4 Extrinsic Camera Matrix [R | t]:")
    print(test_orbit[0]["transform_matrix"])
    print("=" * 65)
