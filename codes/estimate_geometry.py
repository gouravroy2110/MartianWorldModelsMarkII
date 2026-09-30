"""
estimate_geometry.py — Modular Monocular Geometry Extraction.
Uses Depth Anything V2 (Small) to estimate dense depth, and computes analytical
surface normal vectors via camera-space spatial depth gradients.

Outputs saved to target directory:
  - <name>_depth.npy: Raw floating-point depth map
  - <name>_depth_vis.png: 8-bit colormapped depth visualization (inferno)
  - <name>_normals.npy: (H, W, 3) unit normal vectors
  - <name>_normals_vis.png: (H, W, 3) RGB surface normal visualization
"""

import os
import glob
import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


def compute_surface_normals(depth: torch.Tensor, fx: float = 1.0, fy: float = 1.0) -> torch.Tensor:
    """
    Computes camera-space surface normals from dense depth map via spatial gradient cross-products.
    
    Args:
        depth: (H, W) or (1, 1, H, W) tensor of positive depth values.
        fx, fy: Normalized focal lengths.
        
    Returns:
        normals: (H, W, 3) normalized unit normal vectors pointing towards the camera (Z > 0).
    """
    if depth.ndim == 2:
        depth = depth.unsqueeze(0).unsqueeze(0)
    elif depth.ndim == 3:
        depth = depth.unsqueeze(0)
        
    B, C, H, W = depth.shape
    device = depth.device

    # Pixel coordinate grid
    y_coords, x_coords = torch.meshgrid(
        torch.linspace(-1.0, 1.0, H, device=device),
        torch.linspace(-1.0, 1.0, W, device=device),
        indexing="ij"
    )

    # 3D unprojected points in camera space
    Z = depth.squeeze(0).squeeze(0)
    X = x_coords * Z / fx
    Y = y_coords * Z / fy

    # Spatial central differences
    # dX/dx, dY/dx, dZ/dx
    dx_X = torch.gradient(X, dim=1)[0]
    dx_Y = torch.gradient(Y, dim=1)[0]
    dx_Z = torch.gradient(Z, dim=1)[0]
    vec_x = torch.stack([dx_X, dx_Y, dx_Z], dim=-1)

    # dX/dy, dY/dy, dZ/dy
    dy_X = torch.gradient(X, dim=0)[0]
    dy_Y = torch.gradient(Y, dim=0)[0]
    dy_Z = torch.gradient(Z, dim=0)[0]
    vec_y = torch.stack([dy_X, dy_Y, dy_Z], dim=-1)

    # Surface normal = cross(vec_x, vec_y)
    normals = torch.cross(vec_x, vec_y, dim=-1)
    norm = torch.linalg.norm(normals, dim=-1, keepdim=True).clamp(min=1e-8)
    normals = normals / norm

    # Ensure normals consistently face the camera (+Z)
    normals = torch.where(normals[..., 2:3] < 0, -normals, normals)
    return normals


def run_geometry_pipeline(input_dir: str, output_dir: str, model_id: str = "depth-anything/Depth-Anything-V2-Small-hf"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading Depth Anything V2 ({model_id}) on {device}...")
    
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device)
    model.eval()

    os.makedirs(output_dir, exist_ok=True)
    image_paths = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    
    if not image_paths:
        print(f"No PNG images found in {input_dir}")
        return

    print(f"Processing {len(image_paths)} images from {input_dir}...")

    import matplotlib.cm as cm

    for idx, img_path in enumerate(image_paths, 1):
        filename = os.path.splitext(os.path.basename(img_path))[0]
        print(f"[{idx}/{len(image_paths)}] Estimating geometry for: {filename}")
        
        raw_image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = raw_image.size

        inputs = processor(images=raw_image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_depth = outputs.predicted_depth

        # Interpolate to original resolution
        depth_tensor = F.interpolate(
            predicted_depth.unsqueeze(1),
            size=(orig_h, orig_w),
            mode="bicubic",
            align_corners=False
        ).squeeze()

        # Compute surface normals
        normals_tensor = compute_surface_normals(depth_tensor)

        depth_np = depth_tensor.cpu().numpy()
        normals_np = normals_tensor.cpu().numpy()

        # Save raw numpy files
        depth_out_path = os.path.join(output_dir, f"{filename}_depth.npy")
        normals_out_path = os.path.join(output_dir, f"{filename}_normals.npy")
        np.save(depth_out_path, depth_np)
        np.save(normals_out_path, normals_np)

        # Normalize depth for visualization (0 to 255 with inferno colormap)
        d_min, d_max = depth_np.min(), depth_np.max()
        depth_norm = (depth_np - d_min) / (d_max - d_min + 1e-8)
        depth_vis = (cm.inferno(depth_norm)[..., :3] * 255).astype(np.uint8)
        Image.fromarray(depth_vis).save(os.path.join(output_dir, f"{filename}_depth_vis.png"))

        # Map normals from [-1, 1] to [0, 255] RGB visualization
        normals_vis = ((normals_np + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)
        Image.fromarray(normals_vis).save(os.path.join(output_dir, f"{filename}_normals_vis.png"))

        print(f"  -> Saved depth: {depth_out_path} (Range: [{d_min:.2f}, {d_max:.2f}])")
        print(f"  -> Saved normals: {normals_out_path}")

    print(f"Geometry extraction complete. All outputs in: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate depth and surface normals using pretrained models.")
    parser.add_argument("--input_dir", type=str, default="data/nasa_raw", help="Path to input images directory")
    parser.add_argument("--output_dir", type=str, default="output/geometry", help="Path to output directory")
    parser.add_argument("--model_id", type=str, default="depth-anything/Depth-Anything-V2-Small-hf", help="Hugging Face model ID")
    args = parser.parse_args()

    run_geometry_pipeline(args.input_dir, args.output_dir, args.model_id)
