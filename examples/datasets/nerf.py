"""
NeRF synthetic dataset loader for 2DGS training.

This module provides Parser and Dataset classes for loading NeRF synthetic datasets
(e.g., from the NeRF synthetic blender dataset).

Expected directory structure:
    data_dir/
    ├── transforms_train.json  # Training camera poses
    ├── transforms_test.json   # Test camera poses
    ├── points3d.ply or scene_name_vh_clean_2.ply  # Optional point cloud
    └── train/                 # Training images (or images/)
        └── *.png

The transforms.json files should have the standard NeRF format:
{
    "camera_angle_x": float,
    "frames": [
        {
            "file_path": str,
            "transform_matrix": [4x4 matrix],
            ...
        },
        ...
    ]
}
"""

import json
import os
import sys
from typing import Dict, List, Optional, Any, NamedTuple
from plyfile import PlyData, PlyElement
import imageio.v2 as imageio
import numpy as np
import torch
import OpenEXR

# ============================================================
# BasicPointCloud and SH2RGB (from OccamLGS utils)
# ============================================================

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

# SH constant for degree 0
C0 = 0.28209479177387814

def SH2RGB(sh):
    """Convert spherical harmonics to RGB."""
    return sh * C0 + 0.5

def read_exr(path):
    """Read OpenEXR depth map file."""
    import Imath
    exr_file = OpenEXR.InputFile(path)
    header = exr_file.header()
    dw = header['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    float_type = Imath.PixelType(Imath.PixelType.FLOAT)
    depth_str = exr_file.channel('R', float_type)
    depth = np.frombuffer(depth_str, dtype=np.float32).reshape(size[1], size[0])
    exr_file.close()
    return depth

def fetchPly_scannet(path):
    """Load ScanNet point cloud from PLY file."""
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    estimated_normals = np.random.random((len(vertices['x']), 3)) / 255.0
    print("using given point clouds! total "+str(len(vertices['x']))+" points!")
    return BasicPointCloud(points=positions, colors=colors, normals=estimated_normals)

def storePly(path, xyz, rgb):
    """Store point cloud to PLY file."""
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


class Parser:
    """NeRF synthetic dataset parser."""

    def __init__(
        self,
        data_dir: str,
        factor: int = 1,
        normalize: bool = False,
        test_every: int = 8,
    ):
        self.data_dir = data_dir
        self.factor = factor
        self.normalize = normalize
        self.test_every = test_every

        # Load training and test transforms
        train_transforms_path = os.path.join(data_dir, "transforms_train.json")
        test_transforms_path = os.path.join(data_dir, "transforms_test.json")

        if not os.path.exists(train_transforms_path):
            raise ValueError(f"Training transforms file not found: {train_transforms_path}")

        with open(train_transforms_path, "r") as f:
            train_transforms = json.load(f)

        # Check if test transforms file exists
        if os.path.exists(test_transforms_path):
            print(f"[Parser] Loading test transforms from: {test_transforms_path}")
            with open(test_transforms_path, "r") as f:
                test_transforms = json.load(f)
        else:
            print(f"[Parser] Test transforms not found, sampling from train (every {test_every} frames)")
            # Sample test frames from train transforms (LLFF-style)
            train_frames = train_transforms.get("frames", [])
            test_frames = [train_frames[i] for i in range(0, len(train_frames), test_every)]
            # Create test_transforms dict with same structure as train_transforms
            test_transforms = {
                "camera_angle_x": train_transforms.get("camera_angle_x"),
                "fl_x": train_transforms.get("fl_x"),
                "fl_y": train_transforms.get("fl_y"),
                "cx": train_transforms.get("cx"),
                "cy": train_transforms.get("cy"),
                "frames": test_frames
            }
            print(f"[Parser] Sampled {len(test_frames)} test frames from {len(train_frames)} train frames")

        # Extract camera parameters
        self.camera_angle_x = train_transforms.get("camera_angle_x")
        self.fl_x = train_transforms.get("fl_x")
        self.fl_y = train_transforms.get("fl_y")
        self.cx = train_transforms.get("cx")
        self.cy = train_transforms.get("cy")

        # Check for monodepth folder
        self.monodepth_folder = os.path.join(data_dir, "monodepth")
        if not os.path.exists(self.monodepth_folder):
            self.monodepth_folder = ""
            print("[Parser] Monodepth folder not found, depth/normal supervision disabled")
        else:
            print(f"[Parser] Monodepth folder found: {self.monodepth_folder}")

        # Process frames
        self.image_names = []
        self.image_paths = []
        self.camtoworlds = []
        self.camera_ids = []  # All same for synthetic dataset
        self.mono_depth_paths = []  # Mono depth map paths
        self.mono_normal_paths = []  # Mono normal map paths

        # Process training frames
        self._process_frames(train_transforms["frames"], "train")

        # Process test frames
        test_offset = len(self.image_names)
        self._process_frames(test_transforms["frames"], "test", test_offset)

        # Convert to numpy arrays
        self.camtoworlds = np.stack(self.camtoworlds, axis=0)  # [N, 4, 4]
        self.camera_ids = np.array(self.camera_ids)  # [N]
        num_images = len(self.image_names)

        # Build Ks_dict (all same intrinsics for synthetic dataset)
        self.Ks_dict = {}
        self.params_dict = {}
        self.imsize_dict = {}
        self.mask_dict = {}

        # Load first image to get dimensions
        first_image = imageio.imread(self.image_paths[0])[..., :3]
        height, width = first_image.shape[:2]

        # Downsample if factor > 1
        if factor > 1:
            width = width // factor
            height = height // factor

        # Build intrinsics matrix
        if self.fl_x is not None:
            fx = self.fl_x / factor
            fy = self.fl_y / factor if self.fl_y is not None else fx
        elif self.camera_angle_x is not None:
            fx = 0.5 * width / np.tan(0.5 * self.camera_angle_x)
            fy = fx
        else:
            raise ValueError("Either camera_angle_x or fl_x must be specified")

        cx = (self.cx / factor) if self.cx is not None else width / 2.0
        cy = (self.cy / factor) if self.cy is not None else height / 2.0

        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

        # All images share the same camera (camera_id = 0)
        camera_id = 0
        for i in range(num_images):
            self.camera_ids[i] = camera_id

        self.Ks_dict[camera_id] = K
        self.params_dict[camera_id] = np.empty(0, dtype=np.float32)  # No distortion
        self.imsize_dict[camera_id] = (width, height)
        self.mask_dict[camera_id] = None  # No mask

        print(f"[Parser] {num_images} images loaded.")

        # Load or initialize point cloud
        scene_name = os.path.basename(data_dir)

        # Try ScanNet format point cloud
        ply_path = os.path.join(data_dir, scene_name + '_vh_clean_2.ply')
        print(f"[Parser] Looking for point cloud: {ply_path}")

        if os.path.exists(ply_path):
            try:
                pcd = fetchPly_scannet(ply_path)
                self.points = pcd.points  # [N, 3]
                self.points_rgb = (pcd.colors * 255.0).astype(np.uint8)  # [N, 3]
                self.points_err = np.zeros(len(pcd.points), dtype=np.float32)
                self.point_indices = {}  # Empty for now
                print(f"[Parser] Loaded {len(pcd.points)} points from point cloud.")
            except Exception as e:
                print(f"[Parser] Failed to load point cloud: {e}")
                self._init_random_points(scene_name)
        else:
            print(f"[Parser] No point cloud found, initializing with random points.")
            self._init_random_points(scene_name)

        self.transform = np.eye(4)

        # Normalize the world space
        if normalize:
            from .normalize import (
                align_principle_axes,
                similarity_from_cameras,
                transform_cameras,
                transform_points,
            )

            T1 = similarity_from_cameras(self.camtoworlds)
            self.camtoworlds = transform_cameras(T1, self.camtoworlds)
            if self.points is not None:
                self.points = transform_points(T1, self.points)

            T2 = align_principle_axes(self.points) if self.points is not None else np.eye(4)
            self.camtoworlds = transform_cameras(T2, self.camtoworlds)
            if self.points is not None:
                self.points = transform_points(T2, self.points)

            self.transform = T2 @ T1

        # Compute scene scale
        camera_locations = self.camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        self.scene_scale = np.max(dists)

    def _process_frames(self, frames: List[Dict], split: str, offset: int = 0):
        """Process frames from transforms.json."""
        for i, frame in enumerate(frames):
            # Image path
            file_path = frame["file_path"]
            # Try common image folders
            for folder in ["", "train/", "images/"]:
                full_path = os.path.join(self.data_dir, folder, file_path + ".png")
                if os.path.exists(full_path):
                    break
                full_path = os.path.join(self.data_dir, folder, file_path + ".jpg")
                if os.path.exists(full_path):
                    break
            else:
                # Try the raw file_path
                full_path = os.path.join(self.data_dir, file_path)
                if not os.path.exists(full_path):
                    # Try with extension
                    for ext in [".png", ".jpg"]:
                        test_path = full_path + ext
                        if os.path.exists(test_path):
                            full_path = test_path
                            break

            if not os.path.exists(full_path):
                raise ValueError(f"Image not found: {full_path}")

            # Transform matrix (camera-to-world)
            transform_matrix = np.array(frame["transform_matrix"], dtype=np.float32)

            # Convert from OpenGL/Blender (Y up, Z back) to COLMAP (Y down, Z forward)
            transform_matrix[:3, 1:3] *= -1

            image_name = os.path.basename(file_path)

            # Load monodepth depth and normal paths
            mono_depth_path = ""
            mono_normal_path = ""
            if self.monodepth_folder:
                # Extract frame ID from file_path (e.g., "color/0" -> "0")
                frame_id = file_path.split('/')[-1]
                mono_depth_path = os.path.join(self.monodepth_folder, frame_id, "depth.exr")
                mono_normal_path = os.path.join(self.monodepth_folder, frame_id, "normal.png")
                # Set to empty if files don't exist
                if not os.path.exists(mono_depth_path):
                    mono_depth_path = ""
                if not os.path.exists(mono_normal_path):
                    mono_normal_path = ""

            self.image_names.append(image_name)
            self.image_paths.append(full_path)
            self.camtoworlds.append(transform_matrix)
            self.camera_ids.append(0)  # All same camera
            self.mono_depth_paths.append(mono_depth_path)
            self.mono_normal_paths.append(mono_normal_path)

        print(f"[Parser] Loaded {len(frames)} {split} frames.")

    def _init_random_points(self, scene_name: str):
        """Initialize with random point cloud when no SFM points available."""
        num_pts = 100_000
        print(f"[Parser] Generating random point cloud ({num_pts})...")

        # Create random points inside typical bounds [-1.3, 1.3]
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        rgb = SH2RGB(shs)

        self.points = xyz
        self.points_rgb = (rgb * 255.0).astype(np.uint8)
        self.points_err = np.zeros(num_pts, dtype=np.float32)
        self.point_indices = {}

        # Save with correct naming: scene_name + "random.ply"
        ply_path = os.path.join(self.data_dir, scene_name + "random.ply")
        try:
            storePly(ply_path, xyz, rgb * 255.0)
            print(f"[Parser] Saved random points to: {ply_path}")
        except Exception as e:
            print(f"[Parser] Warning: Could not save points: {e}")


class Dataset:
    """NeRF synthetic dataset class."""

    def __init__(
        self,
        parser: Parser,
        split: str = "train",
        patch_size: Optional[int] = None,
        load_depths: bool = False,
    ):
        self.parser = parser
        self.split = split
        self.patch_size = patch_size
        self.load_depths = load_depths

        # Split indices based on test_every
        indices = np.arange(len(self.parser.image_names))
        if split == "train":
            self.indices = indices[indices % self.parser.test_every != 0]
        else:
            self.indices = indices[indices % self.parser.test_every == 0]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        index = self.indices[item]

        # Load image
        image = imageio.imread(self.parser.image_paths[index])[..., :3]

        # Get camera parameters
        camera_id = self.parser.camera_ids[index]
        K = self.parser.Ks_dict[camera_id].copy()
        camtoworld = self.parser.camtoworlds[index].copy()
        mask = self.parser.mask_dict[camera_id]

        # Resize if factor > 1
        if self.parser.factor > 1:
            h, w = image.shape[:2]
            new_h, new_w = self.parser.imsize_dict[camera_id]
            if (h, w) != (new_h, new_w):
                # Use PIL for high-quality resizing
                from PIL import Image
                image = np.array(Image.fromarray(image).resize((new_w, new_h), Image.BICUBIC))

        # Random crop if patch_size is specified
        crop_x, crop_y = None, None
        if self.patch_size is not None:
            h, w = image.shape[:2]
            crop_x = np.random.randint(0, max(w - self.patch_size, 1))
            crop_y = np.random.randint(0, max(h - self.patch_size, 1))
            image = image[crop_y : crop_y + self.patch_size, crop_x : crop_x + self.patch_size]
            K[0, 2] -= crop_x
            K[1, 2] -= crop_y

        # Prepare data dictionary
        data = {
            "K": torch.from_numpy(K).float(),
            "camtoworld": torch.from_numpy(camtoworld).float(),
            "image": torch.from_numpy(image).float(),  # Do not Normalize to [0, 1]
            "image_id": item,
        }

        if mask is not None:
            data["mask"] = torch.from_numpy(mask).bool()

        # Load monodepth depth and normal maps if available
        mono_depth_path = self.parser.mono_depth_paths[index]
        mono_normal_path = self.parser.mono_normal_paths[index]

        if mono_depth_path:
            try:
                depth_map = read_exr(mono_depth_path)
                # Resize depth map if image was resized
                if self.parser.factor > 1:
                    from PIL import Image
                    new_h, new_w = self.parser.imsize_dict[camera_id]
                    depth_map = np.array(Image.fromarray(depth_map).resize((new_w, new_h), Image.BICUBIC))
                # Apply same crop if patch_size is used
                if crop_x is not None and crop_y is not None:
                    depth_map = depth_map[crop_y : crop_y + self.patch_size, crop_x : crop_x + self.patch_size]
                data["mono_depthmap"] = torch.from_numpy(depth_map).float()
            except Exception as e:
                print(f"[Dataset] Warning: Failed to load depth from {mono_depth_path}: {e}")

        if mono_normal_path:
            try:
                normal_map = imageio.imread(mono_normal_path).astype(np.float32) / 255.0
                # Resize normal map if image was resized
                if self.parser.factor > 1:
                    from PIL import Image
                    new_h, new_w = self.parser.imsize_dict[camera_id]
                    normal_map = np.array(Image.fromarray((normal_map * 255).astype(np.uint8)).resize((new_w, new_h), Image.BICUBIC)) / 255.0
                # Apply same crop if patch_size is used
                if crop_x is not None and crop_y is not None:
                    normal_map = normal_map[crop_y : crop_y + self.patch_size, crop_x : crop_x + self.patch_size]
                # Convert from [0,1] to [-1,1] range
                data["mono_normalmap"] = torch.from_numpy(normal_map).float() * 2.0 - 1.0
            except Exception as e:
                print(f"[Dataset] Warning: Failed to load normal from {mono_normal_path}: {e}")

        # Depth loading (not typically available in NeRF synthetic)
        if self.load_depths:
            data["depth"] = torch.zeros(1)  # Placeholder

        return data


def create_random_points(num_pts: int, extent: float = 3.0) -> np.ndarray:
    """Create random 3D points for initialization when no SFM points available."""
    return (np.random.rand(num_pts, 3) * 2 - 1) * extent
