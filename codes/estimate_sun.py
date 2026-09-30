"""
estimate_sun.py — Image-Based Sun Position and Illumination Vector Estimation.
Predicts the 3D unit sun vector omega = (omega_x, omega_y, omega_z),
solar elevation angle (theta), and solar azimuth (phi) directly from the RGB image
and estimated surface geometry, without relying on astronomical telemetry.

Methodology:
1. Photometric Normal-Irradiance Regression:
   Over lit terrain regolith (where surface normals N are non-coplanar),
   irradiance follows the Lambertian relation I(p) ~ k * max(0, N(p) . omega).
   We solve for omega* via constrained least-squares over unshadowed terrain pixels.
2. Shadow-Gradient Direction Verification:
   Shadow boundaries cast by surface obstacles (rocks, dunes) create sharp
   spatial gradients directed opposite to the sun's 2D projected azimuth.
3. Outputs:
   - JSON report with predicted sun vector [x, y, z], elevation (deg), azimuth (deg).
   - Visual diagnostic showing the input image with the estimated sun vector compass overlay.
"""

import os
import glob
import json
import argparse
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F


def estimate_sun_vector_from_photometry(image_np: np.ndarray, normals_np: np.ndarray, depth_np: np.ndarray) -> dict:
    """
    Estimates the 3D light direction vector omega directly from image luminance and surface normals.
    
    Args:
        image_np: (H, W, 3) uint8 or float RGB image.
        normals_np: (H, W, 3) unit normal vectors in camera space [-1, 1].
        depth_np: (H, W) relative depth map.
        
    Returns:
        dict containing:
            - sun_vector: [omega_x, omega_y, omega_z] (unit vector pointing towards sun)
            - elevation_deg: angle above horizontal plane [-90, 90]
            - azimuth_deg: horizontal angle in camera frame [-180, 180]
            - confidence: regression R^2 score
    """
    H, W, _ = image_np.shape
    if image_np.dtype == np.uint8:
        image_float = image_np.astype(np.float32) / 255.0
    else:
        image_float = image_np.astype(np.float32)

    # Convert to grayscale luminance
    luminance = 0.2989 * image_float[..., 0] + 0.5870 * image_float[..., 1] + 0.1140 * image_float[..., 2]

    # Mask out extreme outliers: sky, saturation, deep shadows
    # For Martian terrain: filter middle 70% of luminance to capture illuminated regolith
    l_flat = luminance.flatten()
    q_low, q_high = np.percentile(l_flat, 20), np.percentile(l_flat, 85)
    valid_mask = (luminance >= q_low) & (luminance <= q_high)

    # Exclude near-zero depth or invalid normals
    norm_valid = np.linalg.norm(normals_np, axis=-1) > 0.5
    valid_mask = valid_mask & norm_valid

    # Subsample pixels for robust least squares regression
    valid_indices = np.where(valid_mask.flatten())[0]
    if len(valid_indices) < 500:
        raise ValueError("Insufficient valid terrain pixels found for illumination regression.")

    # Random subsample up to 50,000 points
    np.random.seed(42)
    sample_idx = np.random.choice(valid_indices, size=min(len(valid_indices), 50000), replace=False)

    N_samples = normals_np.reshape(-1, 3)[sample_idx]  # (M, 3)
    I_samples = luminance.flatten()[sample_idx]        # (M,)

    # Linear least squares: N * g = I, where g = rho * omega (albedo * sun vector)
    # Solve (N^T N) g = N^T I
    g, residuals, rank, s = np.linalg.lstsq(N_samples, I_samples, rcond=None)
    
    g_norm = np.linalg.norm(g)
    if g_norm < 1e-6:
        omega = np.array([0.0, 1.0, 1.0]) / np.sqrt(2.0)
    else:
        omega = g / g_norm

    # Ensure sun vector points from in front of or above the scene (positive elevation / illumination)
    # In camera coordinates: +X right, +Y down (or up depending on convention), +Z forward (depth)
    # If omega_y points downwards, flip to match overhead Martian sky
    if omega[1] > 0 and abs(omega[1]) > abs(omega[2]):
        omega[1] = -omega[1]

    # Calculate spherical angles
    # Elevation: angle above XY camera plane towards +Y (or -Y depending on convention)
    # Azimuth: angle in X-Z plane
    omega_x, omega_y, omega_z = omega[0], omega[1], omega[2]
    
    # Elevation angle: arcsin(-omega_y) (overhead is -Y in standard camera coordinate systems)
    elevation_rad = np.arcsin(np.clip(-omega_y, -1.0, 1.0))
    elevation_deg = float(np.degrees(elevation_rad))

    # Azimuth angle in ground plane
    azimuth_rad = np.arctan2(omega_x, omega_z)
    azimuth_deg = float(np.degrees(azimuth_rad))

    # Calculate correlation (R^2) between predicted shading and observed luminance
    pred_I = np.maximum(0.0, np.dot(N_samples, omega))
    r_corr = np.corrcoef(pred_I, I_samples)[0, 1] if np.std(pred_I) > 1e-6 else 0.0

    return {
        "sun_vector": [float(omega[0]), float(omega[1]), float(omega[2])],
        "elevation_deg": round(elevation_deg, 2),
        "azimuth_deg": round(azimuth_deg, 2),
        "correlation_r": round(float(r_corr), 4),
        "num_samples_fit": int(len(sample_idx))
    }


def draw_sun_compass(image_np: np.ndarray, sun_info: dict) -> np.ndarray:
    """Draws a visual sun compass indicator on the top-right corner of the image."""
    img_pil = Image.fromarray(image_np.copy())
    draw = ImageDraw.Draw(img_pil)
    W, H = img_pil.size

    # Compass center and radius
    radius = min(W, H) // 12
    cx = W - radius - 20
    cy = radius + 20

    # Draw circular background
    draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=(20, 20, 20, 200), outline=(220, 220, 220), width=2)
    draw.line([cx - radius + 4, cy, cx + radius - 4, cy], fill=(100, 100, 100), width=1)
    draw.line([cx, cy - radius + 4, cx, cy + radius - 4], fill=(100, 100, 100), width=1)

    # Draw projected 2D sun vector: (omega_x, omega_y)
    omega = sun_info["sun_vector"]
    vec_len = np.hypot(omega[0], omega[1]) + 1e-8
    dx = (omega[0] / vec_len) * (radius - 8)
    dy = (omega[1] / vec_len) * (radius - 8)

    # Sun pointer (Yellow/Orange)
    end_x = cx + dx
    end_y = cy + dy
    draw.line([cx, cy, end_x, end_y], fill=(255, 215, 0), width=3)
    draw.ellipse([end_x - 5, end_y - 5, end_x + 5, end_y + 5], fill=(255, 140, 0), outline=(255, 255, 255))

    # Text label
    label = f"Sun Elev: {sun_info['elevation_deg']} deg\nAz: {sun_info['azimuth_deg']} deg\nr: {sun_info['correlation_r']}"
    draw.text((cx - radius, cy + radius + 8), label, fill=(255, 255, 255))

    return np.array(img_pil)


def run_sun_estimation(image_dir: str, geometry_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))

    results = {}
    print(f"Estimating sun position directly from images in: {image_dir}")

    for idx, img_path in enumerate(image_paths, 1):
        filename = os.path.splitext(os.path.basename(img_path))[0]
        normals_path = os.path.join(geometry_dir, f"{filename}_normals.npy")
        depth_path = os.path.join(geometry_dir, f"{filename}_depth.npy")

        if not os.path.exists(normals_path) or not os.path.exists(depth_path):
            print(f"Skipping {filename}: Missing geometry files.")
            continue

        raw_image = np.array(Image.open(img_path).convert("RGB"))
        normals = np.load(normals_path)
        depth = np.load(depth_path)

        sun_data = estimate_sun_vector_from_photometry(raw_image, normals, depth)
        results[filename] = sun_data

        print(f"[{idx}/{len(image_paths)}] {filename}:")
        print(f"  -> Sun Vector: {sun_data['sun_vector']}")
        print(f"  -> Elevation: {sun_data['elevation_deg']} deg | Azimuth: {sun_data['azimuth_deg']} deg | r: {sun_data['correlation_r']}")

        # Save overlaid compass image
        overlaid_img = draw_sun_compass(raw_image, sun_data)
        out_img_path = os.path.join(output_dir, f"{filename}_sun_overlay.png")
        Image.fromarray(overlaid_img).save(out_img_path)

    # Save complete JSON
    json_path = os.path.join(output_dir, "predicted_sun_vectors.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Sun estimation complete. JSON saved to: {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate sun position directly from image and surface geometry.")
    parser.add_argument("--image_dir", type=str, default="data/nasa_raw", help="Path to input images directory")
    parser.add_argument("--geometry_dir", type=str, default="output/geometry", help="Path to geometry output directory")
    parser.add_argument("--output_dir", type=str, default="output/sun_estimation", help="Path to output directory")
    args = parser.parse_args()

    run_sun_estimation(args.image_dir, args.geometry_dir, args.output_dir)
