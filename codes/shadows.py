"""
shadows.py — Vectorized PyTorch Screen-Space Shadow Ray Marcher.
Implements the screen-space shadow ray-marching algorithm described in OutCast
(Griffiths et al., ECCV 2022) and real-time screen-space directional occlusion.

Input:
  - Dense depth map D: (H, W) or (1, 1, H, W) [Disparity or Metric Depth]
  - 3D Unit Sun vector omega = (omega_x, omega_y, omega_z)

Output:
  - Cast shadow mask S in [0, 1]^(H x W) (1 = in shadow, 0 = fully illuminated)
  - Diagnostic overlay comparing original image with computed shadow boundaries
"""

import os
import glob
import math
import json
import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


def disparity_to_metric_depth(disp_np: np.ndarray, target_mean_distance: float = 2.0) -> np.ndarray:
    """
    Converts Depth Anything V2 / MiDaS disparity to metric camera-space depth Z (in meters).
    DPT models output relative inverse depth (disparity) where closer = larger.
    Metric Z = s / (disp + eps), normalized to typical working distance.
    """
    disp_clean = np.clip(disp_np, a_min=0.05, a_max=None)
    z_raw = 1.0 / disp_clean
    scale = target_mean_distance / np.mean(z_raw)
    return (z_raw * scale).astype(np.float32)


class ScreenSpaceShadowRayMarcher:
    def __init__(
        self,
        num_steps: int = 48,
        step_size: float = 0.025,
        bias: float = 0.015,
        thickness: float = 0.25,
        soft_shadow: bool = True,
        dither: bool = True
    ):
        """
        Args:
            num_steps: Number of ray marching steps along the light vector.
            step_size: Step distance along the 3D ray in metric camera units.
            bias: Surface self-shadowing tolerance epsilon to avoid shadow acne.
            thickness: Maximum occluder depth thickness to prevent back-face false occlusions.
            soft_shadow: Whether to generate soft shadow penumbra via step distance weighting.
            dither: Whether to use pseudo-random stochastic ray step offsets to eliminate banding.
        """
        self.num_steps = num_steps
        self.step_size = step_size
        self.bias = bias
        self.thickness = thickness
        self.soft_shadow = soft_shadow
        self.dither = dither

    def march(
        self,
        depth: torch.Tensor,
        sun_vector: torch.Tensor,
        fov_deg: float = 15.0
    ) -> torch.Tensor:
        """
        Executes vectorized screen-space ray marching for all pixels simultaneously.

        Args:
            depth: (H, W) or (1, 1, H, W) tensor containing metric depth values Z > 0.
            sun_vector: (3,) unit vector pointing towards the light source in camera space.
            fov_deg: Camera horizontal field of view in degrees.

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
        x_grid = x_grid.unsqueeze(0).unsqueeze(0)
        y_grid = y_grid.unsqueeze(0).unsqueeze(0)

        # 3D unprojected point cloud P = (X, Y, Z) in camera space
        Z = depth.clamp(min=1e-3)
        X = (x_grid / fx) * Z
        Y = (y_grid / fy) * Z

        # Stochastic jitter offset per pixel
        if self.dither:
            torch.manual_seed(42)
            dither_offset = torch.rand((1, 1, H, W), device=device) * (self.step_size * 0.75)
        else:
            dither_offset = 0.0

        shadow_accum = torch.zeros((1, 1, H, W), device=device)

        wx = sun_vector[:, 0:1]
        wy = sun_vector[:, 1:2]
        wz = sun_vector[:, 2:3]

        for step_idx in range(1, self.num_steps + 1):
            t = step_idx * self.step_size + dither_offset

            # Sample point along 3D ray towards the sun: R_k = P + t * omega
            R_x = X + t * wx
            R_y = Y + t * wy
            R_z = Z + t * wz

            # Ray must remain in front of the camera plane (Z > 0)
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
            is_occluded = (
                (sampled_depth < (R_z - self.bias)) &
                (sampled_depth > (R_z - self.thickness)) &
                valid_mask
            )

            if self.soft_shadow:
                step_weight = (1.0 - (step_idx / self.num_steps) * 0.45)
                shadow_accum = torch.maximum(shadow_accum, is_occluded.float() * step_weight)
            else:
                shadow_accum = torch.maximum(shadow_accum, is_occluded.float())

        # Spatial 5x5 box smoothing to filter high-frequency dither noise
        kernel = torch.ones((1, 1, 5, 5), device=device) / 25.0
        shadow_mask = F.conv2d(shadow_accum, kernel, padding=2).squeeze()

        return shadow_mask.clamp(0.0, 1.0)


def run_shadow_pipeline(
    image_dir: str,
    geometry_dir: str,
    sun_json_path: str,
    output_dir: str,
    num_steps: int = 48,
    step_size: float = 0.025
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing Dithered Screen-Space Shadow Ray Marcher on {device}...")

    # Load predicted sun vectors
    sun_vectors_dict = {}
    if os.path.exists(sun_json_path):
        with open(sun_json_path, "r", encoding="utf-8") as f:
            sun_vectors_dict = json.load(f)

    marcher = ScreenSpaceShadowRayMarcher(
        num_steps=num_steps,
        step_size=step_size,
        bias=0.015,
        thickness=0.25,
        soft_shadow=True,
        dither=True
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
        depth_disp = np.load(depth_path)
        z_metric = disparity_to_metric_depth(depth_disp, target_mean_distance=2.0)
        depth_tensor = torch.from_numpy(z_metric).float().to(device)

        if filename in sun_vectors_dict:
            raw_omega = sun_vectors_dict[filename]["sun_vector"]
            # Enforce light source from above/behind camera: omega_y < 0, omega_z < 0
            omega_x = float(raw_omega[0])
            omega_y = -abs(float(raw_omega[1]))
            omega_z = -abs(float(raw_omega[2]))
            sun_vec = torch.tensor([omega_x, omega_y, omega_z], dtype=torch.float32)
            print(f"  -> Aligned Light Vector: [{omega_x:.4f}, {omega_y:.4f}, {omega_z:.4f}]")
        else:
            sun_vec = torch.tensor([0.0, -0.30, -0.95], dtype=torch.float32)
            print(f"  -> Default Light Vector: {sun_vec.tolist()}")

        with torch.no_grad():
            shadow_mask = marcher.march(depth_tensor, sun_vec, fov_deg=15.0)

        shadow_np = shadow_mask.cpu().numpy()

        # Save raw mask array
        mask_out_path = os.path.join(output_dir, f"{filename}_shadow_mask.npy")
        np.save(mask_out_path, shadow_np)

        # Save binary/soft mask PNG (black = shadow, white = lit)
        mask_vis = ((1.0 - shadow_np) * 255.0).clip(0, 255).astype(np.uint8)
        Image.fromarray(mask_vis).save(os.path.join(output_dir, f"{filename}_shadow_mask.png"))

        # Save diagnostic overlay: Original image with shadow overlay tinted in cool cyan/blue
        orig_img = Image.open(img_path).convert("RGB")
        orig_np = np.array(orig_img).astype(np.float32)

        overlay_np = orig_np.copy()
        overlay_np[..., 0] = overlay_np[..., 0] * (1.0 - 0.7 * shadow_np)
        overlay_np[..., 1] = overlay_np[..., 1] * (1.0 - 0.4 * shadow_np)
        overlay_np[..., 2] = overlay_np[..., 2] * (1.0 + 0.3 * shadow_np)
        overlay_vis = overlay_np.clip(0, 255).astype(np.uint8)
        overlay_out_path = os.path.join(output_dir, f"{filename}_shadow_overlay.jpg")
        Image.fromarray(overlay_vis).save(overlay_out_path, quality=92)

        shadow_ratio = float((shadow_np > 0.3).mean() * 100.0)
        print(f"  -> Saved shadow mask: {mask_out_path} (Shadowed Area: {shadow_ratio:.1f}%)")
        print(f"  -> Saved shadow overlay: {overlay_out_path}")

    print(f"All shadow computations complete. Outputs saved in: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Screen-Space Shadow Ray Marcher.")
    parser.add_argument("--image_dir", type=str, default="data/nasa_raw")
    parser.add_argument("--geometry_dir", type=str, default="output/geometry")
    parser.add_argument("--sun_json", type=str, default="output/sun_estimation/predicted_sun_vectors.json")
    parser.add_argument("--output_dir", type=str, default="output/shadows")
    parser.add_argument("--num_steps", type=int, default=48)
    parser.add_argument("--step_size", type=float, default=0.025)
    args = parser.parse_args()

    run_shadow_pipeline(
        args.image_dir,
        args.geometry_dir,
        args.sun_json,
        args.output_dir,
        args.num_steps,
        args.step_size
    )
