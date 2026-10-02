"""
shadows.py — Vectorized PyTorch Screen-Space Shadow Ray Marcher.
Implements the screen-space shadow ray-marching algorithm described in OutCast
(Griffiths et al., ECCV 2022) and real-time screen-space directional occlusion.

Input:
  - Dense depth map D: (H, W) or (1, 1, H, W)
  - 3D Unit Sun vector omega = (omega_x, omega_y, omega_z) or (elevation, azimuth)

Output:
  - Cast shadow mask S in [0, 1]^(H x W) (1 = in shadow, 0 = fully illuminated)
  - Diagnostic overlay comparing original image with computed shadow boundaries
"""

import os
import glob
import math
import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


def spherical_to_cartesian_sun(elevation_deg: float, azimuth_deg: float) -> torch.Tensor:
    """
    Converts spherical solar elevation and azimuth (in camera coordinates) to 3D unit vector.
    Elevation: angle above horizontal plane (+Y overhead, or -Y depending on camera axes).
    Azimuth: angle in camera horizontal plane (X-Z).
    """
    el_rad = math.radians(elevation_deg)
    az_rad = math.radians(azimuth_deg)

    # In camera coordinates: +X right, -Y up/overhead, +Z forward
    # Elevation above horizon corresponds to -Y direction
    omega_y = -math.sin(el_rad)
    omega_x = math.cos(el_rad) * math.sin(az_rad)
    omega_z = math.cos(el_rad) * math.cos(az_rad)

    vec = torch.tensor([omega_x, omega_y, omega_z], dtype=torch.float32)
    return vec / torch.norm(vec).clamp(min=1e-8)


class ScreenSpaceShadowRayMarcher:
    def __init__(
        self,
        num_steps: int = 48,
        step_size: float = 0.04,
        bias: float = 0.015,
        thickness: float = 0.8,
        soft_shadow: bool = True
    ):
        """
        Args:
            num_steps: Number of ray marching steps along the light vector.
            step_size: Step distance along the 3D ray in camera units.
            bias: Surface self-shadowing tolerance epsilon.
            thickness: Maximum occluder depth thickness to prevent back-face false occlusions.
            soft_shadow: Whether to generate soft shadow penumbra via step distance weighting.
        """
        self.num_steps = num_steps
        self.step_size = step_size
        self.bias = bias
        self.thickness = thickness
        self.soft_shadow = soft_shadow

    def march(
        self,
        depth: torch.Tensor,
        sun_vector: torch.Tensor,
        fov_deg: float = 15.0
    ) -> torch.Tensor:
        """
        Executes vectorized screen-space ray marching for all pixels simultaneously.

        Args:
            depth: (H, W) or (1, 1, H, W) tensor containing depth values > 0.
            sun_vector: (3,) unit vector pointing towards the light source.
            fov_deg: Camera horizontal field of view in degrees (Mastcam-Z ~15 deg at zoom).

        Returns:
            shadow_mask: (H, W) tensor with values in [0, 1] (1 = shadowed, 0 = lit).
        """
        if depth.ndim == 2:
            depth = depth.unsqueeze(0).unsqueeze(0)
        elif depth.ndim == 3:
            depth = depth.unsqueeze(0)

        device = depth.device
        B, C, H, W = depth.shape
        sun_vector = sun_vector.to(device).view(1, 3, 1, 1)
        sun_vector = sun_vector / torch.norm(sun_vector, dim=1, keepdim=True).clamp(min=1e-8)

        # Camera intrinsics (normalized screen space [-1, 1])
        aspect_ratio = W / H
        half_fov_rad = math.radians(fov_deg / 2.0)
        fx = 1.0 / math.tan(half_fov_rad)
        fy = fx * aspect_ratio

        # Coordinate grids in [-1, 1]
        y_grid, x_grid = torch.meshgrid(
            torch.linspace(-1.0, 1.0, H, device=device),
            torch.linspace(-1.0, 1.0, W, device=device),
            indexing="ij"
        )
        x_grid = x_grid.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        y_grid = y_grid.unsqueeze(0).unsqueeze(0)

        # 3D unprojected point cloud P = (X, Y, Z) in camera space
        Z = depth.clamp(min=1e-3)
        X = (x_grid / fx) * Z
        Y = (y_grid / fy) * Z
        P = torch.cat([X, Y, Z], dim=1)  # (1, 3, H, W)

        # Accumulator for occlusion
        if self.soft_shadow:
            shadow_accum = torch.zeros((1, 1, H, W), device=device)
        else:
            shadow_accum = torch.zeros((1, 1, H, W), dtype=torch.bool, device=device)

        # Vectorized Marching Loop
        wx = sun_vector[:, 0:1]
        wy = sun_vector[:, 1:2]
        wz = sun_vector[:, 2:3]

        for step_idx in range(1, self.num_steps + 1):
            t = step_idx * self.step_size

            # Sample point along 3D ray: R_k = P + t * omega
            R_x = X + t * wx
            R_y = Y + t * wy
            R_z = Z + t * wz

            # Prevent projecting rays that pass behind the camera (Z <= 0)
            valid_z = R_z > 1e-3

            # Project R_k back onto normalized screen coordinates [-1, 1]
            proj_x = (R_x / R_z.clamp(min=1e-3)) * fx
            proj_y = (R_y / R_z.clamp(min=1e-3)) * fy

            # Mask coordinates inside the image boundary [-1, 1]
            inside_screen = (proj_x >= -1.0) & (proj_x <= 1.0) & (proj_y >= -1.0) & (proj_y <= 1.0)
            valid_mask = valid_z & inside_screen

            # Format for torch.nn.functional.grid_sample: (B, H, W, 2)
            grid = torch.cat([proj_x, proj_y], dim=1).permute(0, 2, 3, 1)

            # Sample terrain depth at the projected screen position
            sampled_depth = F.grid_sample(
                depth,
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True
            )

            # Occlusion Condition:
            # The terrain surface at (proj_x, proj_y) is closer than the ray (sampled_depth < R_z - bias)
            # but not excessively far (within occluder thickness threshold)
            is_occluded = (
                (sampled_depth < (R_z - self.bias)) &
                (sampled_depth > (R_z - self.thickness)) &
                valid_mask
            )

            if self.soft_shadow:
                # Soft penumbra weighting: nearby occluders cast sharper shadows, distant cast softer
                step_weight = (1.0 - (step_idx / self.num_steps) * 0.5)
                shadow_accum = torch.maximum(shadow_accum, is_occluded.float() * step_weight)
            else:
                shadow_accum = shadow_accum | is_occluded

        shadow_mask = shadow_accum.squeeze(0).squeeze(0)
        if not self.soft_shadow:
            shadow_mask = shadow_mask.float()

        # Slight spatial bilateral/box smoothing for natural contact shadow falloff
        shadow_mask = shadow_mask.unsqueeze(0).unsqueeze(0)
        kernel = torch.ones((1, 1, 3, 3), device=device) / 9.0
        shadow_mask = F.conv2d(shadow_mask, kernel, padding=1).squeeze()

        return shadow_mask.clamp(0.0, 1.0)


def run_shadow_pipeline(
    image_dir: str,
    geometry_dir: str,
    sun_json_path: str,
    output_dir: str,
    num_steps: int = 48,
    step_size: float = 0.04
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing Screen-Space Shadow Ray Marcher on {device}...")

    # Load predicted or ground-truth sun vectors
    sun_vectors_dict = {}
    if os.path.exists(sun_json_path):
        import json
        with open(sun_json_path, "r", encoding="utf-8") as f:
            sun_vectors_dict = json.load(f)

    marcher = ScreenSpaceShadowRayMarcher(
        num_steps=num_steps,
        step_size=step_size,
        bias=0.015,
        thickness=0.8,
        soft_shadow=True
    )

    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
    if not image_paths:
        print(f"No PNG files found in {image_dir}")
        return

    for idx, img_path in enumerate(image_paths, 1):
        filename = os.path.splitext(os.path.basename(img_path))[0]
        depth_path = os.path.join(geometry_dir, f"{filename}_depth.npy")

        if not os.path.exists(depth_path):
            print(f"Skipping {filename}: Missing depth map at {depth_path}")
            continue

        print(f"[{idx}/{len(image_paths)}] Processing shadow ray-marching for: {filename}")
        depth_np = np.load(depth_path)
        depth_tensor = torch.from_numpy(depth_np).float().to(device)

        # Retrieve estimated sun vector or use default overhead sun
        if filename in sun_vectors_dict:
            omega_vals = sun_vectors_dict[filename]["sun_vector"]
            sun_vec = torch.tensor(omega_vals, dtype=torch.float32)
            print(f"  -> Using Estimated Sun Vector: {omega_vals}")
        else:
            # Default overhead illumination (elevation 30 deg, azimuth 0 deg)
            sun_vec = spherical_to_cartesian_sun(30.0, 0.0)
            print(f"  -> Using Default Overhead Sun Vector: {sun_vec.tolist()}")

        with torch.no_grad():
            shadow_mask = marcher.march(depth_tensor, sun_vec, fov_deg=15.0)

        shadow_np = shadow_mask.cpu().numpy()

        # Save raw shadow mask
        mask_out_path = os.path.join(output_dir, f"{filename}_shadow_mask.npy")
        np.save(mask_out_path, shadow_np)

        # Save visual mask (black = shadow, white = lit)
        mask_vis = ((1.0 - shadow_np) * 255.0).clip(0, 255).astype(np.uint8)
        Image.fromarray(mask_vis).save(os.path.join(output_dir, f"{filename}_shadow_mask.png"))

        # Save diagnostic overlay: Original image with shadow overlay tinted in blue/cyan
        orig_img = Image.open(img_path).convert("RGB")
        orig_np = np.array(orig_img).astype(np.float32)

        # Tint shadow regions
        overlay_np = orig_np.copy()
        shadow_3ch = np.repeat(shadow_np[..., np.newaxis], 3, axis=-1)
        # Darken shadow areas and tint with cool ambient skylight
        overlay_np = overlay_np * (1.0 - 0.45 * shadow_3ch)
        overlay_vis = overlay_np.clip(0, 255).astype(np.uint8)
        Image.fromarray(overlay_vis).save(os.path.join(output_dir, f"{filename}_shadow_overlay.jpg"))

        shadow_ratio = float((shadow_np > 0.4).mean() * 100.0)
        print(f"  -> Saved shadow mask: {mask_out_path} (Shadowed Area: {shadow_ratio:.1f}%)")

    print(f"Shadow computation complete. Outputs in: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Screen-Space Shadow Ray Marcher.")
    parser.add_argument("--image_dir", type=str, default="data/nasa_raw")
    parser.add_argument("--geometry_dir", type=str, default="output/geometry")
    parser.add_argument("--sun_json", type=str, default="output/sun_estimation/predicted_sun_vectors.json")
    parser.add_argument("--output_dir", type=str, default="output/shadows")
    parser.add_argument("--num_steps", type=int, default=48)
    parser.add_argument("--step_size", type=float, default=0.04)
    args = parser.parse_args()

    run_shadow_pipeline(
        args.image_dir,
        args.geometry_dir,
        args.sun_json,
        args.output_dir,
        args.num_steps,
        args.step_size
    )
