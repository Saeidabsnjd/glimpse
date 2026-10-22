"""
glimpse: Automated Blender Synthetic Dataset Engine
Executes headless within Blender 5.x to render paired multi-view datasets:
- Ground Truth: Cycles path-traced at 512+ spp with Subsurface Scattering
- Draft Input: Cycles path-traced at 2 spp
- G-Buffers: Metric Depth (Z) and Camera-Space Surface Normals
- Camera Metadata: transforms.json (NeRF / 3DGS format)
"""

import sys
import os
import json
import math
import argparse
from pathlib import Path

# Add project root to sys.path so we can import our camera_orbit module
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import bpy
import mathutils
from src.data.camera_orbit import generate_hemisphere_orbit


def parse_args():
    """
    Parses command line arguments passed after Blender's '--' separator.
    Example: blender -b -P dataset_engine.py -- --output_dir ./data/run1
    """
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="glimpse Blender Synthetic Dataset Engine")
    parser.add_argument("--output_dir", type=str, default="data/sample_dataset", help="Output directory path")
    parser.add_argument("--samples_gt", type=int, default=512, help="Cycles samples for Ground Truth")
    parser.add_argument("--samples_draft", type=int, default=2, help="Cycles samples for Draft Input")
    parser.add_argument("--n_azimuth", type=int, default=12, help="Horizontal orbit steps (default: 12)")
    parser.add_argument("--n_elevation", type=int, default=4, help="Vertical elevation steps (default: 4)")
    parser.add_argument("--camera_radius", type=float, default=1.6, help="Distance from object in meters")
    parser.add_argument("--resolution", type=int, default=512, help="Image width and height in pixels")
    parser.add_argument("--asset_path", type=str, default=None, help="Optional path to 3D asset (.obj, .fbx, .glb)")
    
    return parser.parse_args(argv)


def configure_optix_gpu():
    """
    Configures Blender Cycles GPU compute.
    - Uses OPTIX on RTX GPUs (e.g., RTX 3060 Ti / RTX 4090) with hardware RT cores.
    - Uses CUDA on GTX GPUs (e.g., GTX 1660 Ti) which lack dedicated RT cores.
    """
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'GPU'

    cycles_prefs = bpy.context.preferences.addons['cycles'].preferences
    
    # Query all available GPU devices
    cycles_prefs.compute_device_type = 'CUDA'
    cycles_prefs.get_devices()
    cuda_devices = [d for d in cycles_prefs.devices if d.type == 'CUDA']
    
    has_rtx = any("RTX" in d.name.upper() for d in cuda_devices)

    if has_rtx:
        # Use OptiX for hardware ray tracing on RTX GPUs
        cycles_prefs.compute_device_type = 'OPTIX'
        cycles_prefs.get_devices()
        for device in cycles_prefs.devices:
            if device.type == 'OPTIX':
                device.use = True
                print(f"[glimpse Engine] Enabled OptiX Hardware RT: {device.name}")
    else:
        # Use CUDA for GTX and non-RTX NVIDIA GPUs
        cycles_prefs.compute_device_type = 'CUDA'
        cycles_prefs.get_devices()
        for device in cycles_prefs.devices:
            if device.type == 'CUDA':
                device.use = True
                print(f"[glimpse Engine] Enabled CUDA GPU Compute: {device.name}")


def create_calibrated_sss_material(name="Skin_SSS_Material"):
    """
    Creates a photorealistic translucent skin material using Blender's Principled BSDF.
    
    Subsurface Scattering Parameters:
        - Subsurface Weight: 0.85 (translucency strength)
        - Subsurface Radius: (1.0, 0.2, 0.1) (Red penetrates 10x deeper than Blue)
        - Subsurface Scale: 0.05 meters (skin layer depth)
        - Subsurface IOR: 1.4 (Human epidermis index of refraction)
    """
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    nodes.clear()

    # Output node
    node_out = nodes.new(type='ShaderNodeOutputMaterial')
    node_out.location = (400, 0)

    # Principled BSDF node
    bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    bsdf.location = (0, 0)

    # Human skin base tone (soft Caucasian / Mediterranean skin tone)
    bsdf.inputs['Base Color'].default_value = (0.82, 0.62, 0.52, 1.0)
    bsdf.inputs['Roughness'].default_value = 0.35

    # Calibrated Subsurface Scattering inputs (Blender 4.x / 5.x socket names)
    if 'Subsurface Weight' in bsdf.inputs:
        bsdf.inputs['Subsurface Weight'].default_value = 0.85
    elif 'Subsurface' in bsdf.inputs:
        bsdf.inputs['Subsurface'].default_value = 0.85

    if 'Subsurface Radius' in bsdf.inputs:
        bsdf.inputs['Subsurface Radius'].default_value = (1.0, 0.2, 0.1)

    if 'Subsurface Scale' in bsdf.inputs:
        bsdf.inputs['Subsurface Scale'].default_value = 0.005

    if 'Subsurface IOR' in bsdf.inputs:
        bsdf.inputs['Subsurface IOR'].default_value = 1.4

    links.new(bsdf.outputs['BSDF'], node_out.inputs['Surface'])

    return mat


def setup_scene(asset_path=None):
    """
    Cleans the default scene, creates camera and studio lights,
    and loads the 3D model (or generates a high-poly test bust).
    """
    # Delete default startup objects (Cube, default Light, default Camera)
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    # Create Studio Lighting (Key Light, Fill Light, Rim Light)
    # 1. Key light (Warm sun)
    key_light_data = bpy.data.lights.new(name="Key_Light", type='POINT')
    key_light_data.energy = 500.0
    key_light_data.color = (1.0, 0.95, 0.9)
    key_light_data.shadow_soft_size = 0.1
    key_light = bpy.data.objects.new(name="Key_Light", object_data=key_light_data)
    key_light.location = (2.0, -2.0, 2.5)
    bpy.context.collection.objects.link(key_light)

    # 2. Rim/Back light (Cool light for SSS edge translucency)
    rim_light_data = bpy.data.lights.new(name="Rim_Light", type='POINT')
    rim_light_data.energy = 700.0
    rim_light_data.color = (0.85, 0.92, 1.0)
    rim_light = bpy.data.objects.new(name="Rim_Light", object_data=rim_light_data)
    rim_light.location = (-1.5, 2.0, 1.5)
    bpy.context.collection.objects.link(rim_light)

    # 3. Soft ambient fill
    bpy.context.scene.world.use_nodes = True
    bg_node = bpy.context.scene.world.node_tree.nodes.get("Background")
    if bg_node:
        bg_node.inputs['Color'].default_value = (0.05, 0.05, 0.06, 1.0)
        bg_node.inputs['Strength'].default_value = 0.4

    # Load 3D Asset or generate test bust
    if asset_path and os.path.exists(asset_path):
        ext = os.path.splitext(asset_path)[1].lower()
        if ext == '.obj':
            bpy.ops.wm.obj_import(filepath=asset_path)
        elif ext in ['.glb', '.gltf']:
            bpy.ops.import_scene.gltf(filepath=asset_path)
        elif ext == '.fbx':
            bpy.ops.import_scene.fbx(filepath=asset_path)
        obj = bpy.context.selected_objects[0]
    else:
        # Generate test bust (Suzanne monkey head with Subdivision Surface)
        bpy.ops.mesh.primitive_monkey_add(size=1.0, location=(0, 0, 0))
        obj = bpy.context.active_object
        obj.rotation_euler = (0.0, 0.0, math.radians(180))

        # Add Subdivision Surface modifier for smooth curved skin
        subdiv = obj.modifiers.new(name="Subsurf", type='SUBSURF')
        subdiv.levels = 2
        subdiv.render_levels = 2
        bpy.ops.object.shade_smooth()

    # Assign the calibrated SSS material
    sss_mat = create_calibrated_sss_material()
    if obj.data.materials:
        obj.data.materials[0] = sss_mat
    else:
        obj.data.materials.append(sss_mat)

    # Create virtual camera
    cam_data = bpy.data.cameras.new(name="glimpse_Camera")
    cam_data.lens = 50.0  # 50mm portrait lens
    cam_data.sensor_width = 36.0  # 35mm full frame sensor
    cam_obj = bpy.data.objects.new(name="glimpse_Camera", object_data=cam_data)
    bpy.context.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj

    return cam_obj, cam_data


def setup_compositor_nodes(output_dir):
    """
    Configures the Blender Compositor to export:
    - Normal Pass (Camera Space)
    - Depth Pass (Metric Z)
    """
    scene = bpy.context.scene
    view_layer = scene.view_layers["ViewLayer"]

    # Enable G-buffer passes in Cycles
    view_layer.use_pass_z = True
    view_layer.use_pass_normal = True

    # Support Blender 5.x compositing_node_group vs Blender 4.x scene.node_tree
    if hasattr(scene, "compositing_node_group"):
        if scene.compositing_node_group is None:
            scene.compositing_node_group = bpy.data.node_groups.new("Compositor", "CompositorNodeTree")
        tree = scene.compositing_node_group
    elif hasattr(scene, "node_tree") and scene.node_tree is not None:
        tree = scene.node_tree
    else:
        scene.use_nodes = True
        tree = scene.node_tree

    tree.nodes.clear()

    # Render Layers Input
    render_layers = tree.nodes.new(type='CompositorNodeRLayers')
    render_layers.location = (0, 0)

    # Create separate output directories for depth and normal
    depth_dir = str(Path(output_dir).resolve() / "depth")
    normal_dir = str(Path(output_dir).resolve() / "normal")
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(normal_dir, exist_ok=True)

    # 1. Depth Output Node
    depth_node = tree.nodes.new(type='CompositorNodeOutputFile')
    depth_node.location = (400, 100)

    # 2. Normal Output Node
    normal_node = tree.nodes.new(type='CompositorNodeOutputFile')
    normal_node.location = (400, -100)

    if hasattr(depth_node, "file_output_items"):
        # Blender 5.x API
        depth_node.directory = depth_dir
        depth_node.format.file_format = 'OPEN_EXR_MULTILAYER'
        depth_node.format.color_depth = '32'
        depth_node.file_output_items.clear()
        depth_node.file_output_items.new('RGBA', 'depth')
        tree.links.new(render_layers.outputs['Depth'], depth_node.inputs['depth'])

        normal_node.directory = normal_dir
        normal_node.format.file_format = 'OPEN_EXR_MULTILAYER'
        normal_node.format.color_depth = '32'
        normal_node.file_output_items.clear()
        normal_node.file_output_items.new('RGBA', 'normal')
        tree.links.new(render_layers.outputs['Normal'], normal_node.inputs['normal'])
    else:
        # Blender 4.x API fallback
        depth_node.base_path = depth_dir
        depth_node.format.file_format = 'OPEN_EXR'
        depth_node.format.color_depth = '32'
        depth_node.file_slots.clear()
        depth_node.file_slots.new("depth")
        tree.links.new(render_layers.outputs['Depth'], depth_node.inputs['depth'])

        normal_node.base_path = normal_dir
        normal_node.format.file_format = 'OPEN_EXR'
        normal_node.format.color_depth = '32'
        normal_node.file_slots.clear()
        normal_node.file_slots.new("normal")
        tree.links.new(render_layers.outputs['Normal'], normal_node.inputs['normal'])

    return depth_node, normal_node


def run_dataset_generation():
    """
    Main loop executing multi-view orbit rendering.
    """
    args = parse_args()
    print("=" * 65)
    print("glimpse: Automated Blender Dataset Engine")
    print("=" * 65)
    print(f"Output directory : {args.output_dir}")
    print(f"Ground Truth spp : {args.samples_gt}")
    print(f"Draft Input spp  : {args.samples_draft}")
    print(f"Resolution       : {args.resolution}x{args.resolution}")
    print("=" * 65)

    # Ensure output directories exist
    out_dir = Path(args.output_dir).resolve()
    gt_dir = out_dir / "ground_truth"
    draft_dir = out_dir / "draft"
    gt_dir.mkdir(parents=True, exist_ok=True)
    draft_dir.mkdir(parents=True, exist_ok=True)

    # Configure hardware & scene
    configure_optix_gpu()
    cam_obj, cam_data = setup_scene(args.asset_path)
    depth_out_node, normal_out_node = setup_compositor_nodes(str(out_dir))

    # Configure resolution
    scene = bpy.context.scene
    scene.render.resolution_x = args.resolution
    scene.render.resolution_y = args.resolution
    scene.render.resolution_percentage = 100

    # Calculate camera intrinsic field of view (FOV)
    fov_rad = cam_data.angle_x
    fx = (args.resolution / 2.0) / math.tan(fov_rad / 2.0)
    fy = fx
    cx = args.resolution / 2.0
    cy = args.resolution / 2.0

    # Generate the hemisphere orbit viewpoints
    viewpoints = generate_hemisphere_orbit(
        radius=args.camera_radius,
        n_azimuth=args.n_azimuth,
        n_elevation=args.n_elevation
    )
    print(f"\n[glimpse Engine] Generated {len(viewpoints)} viewpoints over hemisphere.")

    metadata = {
        "camera_angle_x": fov_rad,
        "focal_length_x": fx,
        "focal_length_y": fy,
        "principal_point_x": cx,
        "principal_point_y": cy,
        "samples_ground_truth": args.samples_gt,
        "samples_draft": args.samples_draft,
        "frames": []
    }

    # Main Rendering Loop
    for idx, vp in enumerate(viewpoints):
        frame_name = f"view_{idx:03d}"
        print(f"\nRendering [{idx+1}/{len(viewpoints)}]: {frame_name} (Az: {vp['azimuth_deg']:.1f}°, El: {vp['elevation_deg']:.1f}°)")

        # 1. Position and orient the camera
        mat4 = vp['transform_matrix']
        cam_obj.matrix_world = mathutils.Matrix(mat4)

        # 2. Render Ground Truth (clean reference, e.g. 512 samples)
        scene.cycles.samples = args.samples_gt
        scene.cycles.use_denoising = True
        scene.render.filepath = str(gt_dir / f"{frame_name}.png")

        if hasattr(depth_out_node, "file_name"):
            depth_out_node.file_name = frame_name
            normal_out_node.file_name = frame_name
            depth_file = f"depth/{frame_name}.exr"
            normal_file = f"normal/{frame_name}.exr"
        else:
            depth_out_node.file_slots["depth"].path = f"{frame_name}_"
            normal_out_node.file_slots["normal"].path = f"{frame_name}_"
            depth_file = f"depth/{frame_name}_0001.exr"
            normal_file = f"normal/{frame_name}_0001.exr"

        depth_out_node.mute = False
        normal_out_node.mute = False
        bpy.ops.render.render(write_still=True)

        # 3. Render Draft Input (raw noisy Monte Carlo input, e.g. 2 samples)
        scene.cycles.samples = args.samples_draft
        scene.cycles.use_denoising = False
        scene.render.filepath = str(draft_dir / f"{frame_name}.png")
        # Mute file output nodes so we do not re-write G-buffers
        depth_out_node.mute = True
        normal_out_node.mute = True
        bpy.ops.render.render(write_still=True)
        depth_out_node.mute = False
        normal_out_node.mute = False

        # 4. Append metadata frame
        metadata["frames"].append({
            "file_path_gt": f"ground_truth/{frame_name}.png",
            "file_path_draft": f"draft/{frame_name}.png",
            "file_path_depth": depth_file,
            "file_path_normal": normal_file,
            "transform_matrix": mat4.tolist(),
            "azimuth_deg": vp['azimuth_deg'],
            "elevation_deg": vp['elevation_deg']
        })

    # Save transforms.json
    transforms_path = out_dir / "transforms.json"
    with open(transforms_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[glimpse Engine] Dataset generation COMPLETE!")
    print(f"[glimpse Engine] Saved transforms metadata to: {transforms_path}")


if __name__ == "__main__":
    run_dataset_generation()
