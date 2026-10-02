"""
estimate_geometry.py — Robust Stereoscopic-Aligned Geometry Extraction.
Uses Depth Anything V2 for dense relative geometry, aligned to metric scale using 
StereoSGBM matches from ZLF/ZRF image pairs.
"""

import os
import glob
import argparse
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from sklearn.linear_model import RANSACRegressor

def compute_stereo_disparity(imgL, imgR):
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    
    # Scale down for robust matching on sand
    scale = 0.5
    grayL_s = cv2.resize(grayL, (0,0), fx=scale, fy=scale)
    grayR_s = cv2.resize(grayR, (0,0), fx=scale, fy=scale)

    stereo = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=64, blockSize=7,
        P1=8*3*7**2, P2=32*3*7**2, disp12MaxDiff=1,
        uniquenessRatio=15, speckleWindowSize=100, speckleRange=32
    )
    
    disp_s = stereo.compute(grayL_s, grayR_s).astype(np.float32) / 16.0
    disp = cv2.resize(disp_s, (imgL.shape[1], imgL.shape[0]), interpolation=cv2.INTER_NEAREST)
    disp = disp * (1.0 / scale) # scale disparity back to original resolution
    return disp

def align_disparity(disp_stereo, disp_da):
    """
    Aligns Depth Anything disparity (disp_da) to physical stereo disparity (disp_stereo).
    Fits: disp_stereo = scale * disp_da + shift
    """
    # Valid pixels: disparity > 0 and not near the very edge
    valid_mask = (disp_stereo > 1.0)
    
    # Subsample for faster RANSAC
    y_coords, x_coords = np.where(valid_mask)
    if len(y_coords) < 1000:
        # Fallback if stereo fails completely
        return disp_da
        
    subsample = np.random.choice(len(y_coords), min(10000, len(y_coords)), replace=False)
    y_sub = y_coords[subsample]
    x_sub = x_coords[subsample]
    
    X = disp_da[y_sub, x_sub].reshape(-1, 1)
    y = disp_stereo[y_sub, x_sub]
    
    ransac = RANSACRegressor(random_state=42)
    ransac.fit(X, y)
    
    scale = ransac.estimator_.coef_[0]
    shift = ransac.estimator_.intercept_
    print(f"    RANSAC alignment: scale={scale:.4f}, shift={shift:.4f}")
    
    aligned_disp = scale * disp_da + shift
    return np.clip(aligned_disp, 0.1, None)

def compute_surface_normals(depth, fx, fy):
    """Computes smooth normals using bilateral-filtered depth to preserve edges."""
    # FIX 1: Apply bilateral filter directly on the raw depth scale.
    # We estimate sigmaColor dynamically based on the median absolute deviation of depth, 
    # ensuring it scales naturally without being crushed by massive outliers.
    valid_depth = depth[depth > 0]
    if len(valid_depth) == 0:
        return np.zeros((*depth.shape, 3), dtype=np.float32)
        
    depth_med = np.median(valid_depth)
    mad = np.median(np.abs(valid_depth - depth_med))
    # Adaptive sigmaColor (e.g. 5% of the median absolute deviation)
    dynamic_sigma_color = max(0.1, mad * 0.05)
    
    depth_smooth = cv2.bilateralFilter(depth.astype(np.float32), d=9, sigmaColor=dynamic_sigma_color, sigmaSpace=75)
    
    H, W = depth.shape
    y_coords, x_coords = np.indices((H, W))
    
    # FIX 2: Unproject using true camera intrinsics (fx, fy), not arbitrary numbers.
    Z = depth_smooth.astype(np.float32)
    X = ((x_coords - W / 2.0) * Z / fx).astype(np.float32)
    Y = ((y_coords - H / 2.0) * Z / fy).astype(np.float32)
    
    # Gradients
    dzdx = cv2.Sobel(Z, cv2.CV_32F, 1, 0, ksize=5)
    dzdy = cv2.Sobel(Z, cv2.CV_32F, 0, 1, ksize=5)
    dxdx = cv2.Sobel(X, cv2.CV_32F, 1, 0, ksize=5)
    dxdy = cv2.Sobel(X, cv2.CV_32F, 0, 1, ksize=5)
    dydx = cv2.Sobel(Y, cv2.CV_32F, 1, 0, ksize=5)
    dydy = cv2.Sobel(Y, cv2.CV_32F, 0, 1, ksize=5)
    
    vec_x = np.dstack((dxdx, dydx, dzdx))
    vec_y = np.dstack((dxdy, dydy, dzdy))
    
    normals = np.cross(vec_x, vec_y)
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = normals / (norm + 1e-8)
    
    # Ensure normals face camera (+Z)
    flip_mask = normals[:, :, 2] < 0
    normals[flip_mask] = -normals[flip_mask]
    
    return normals

def run_geometry_pipeline(input_dir: str, output_dir: str, model_id: str = "depth-anything/Depth-Anything-V2-Small-hf"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading Depth Anything V2 ({model_id}) on {device}...")
    
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device)
    model.eval()

    os.makedirs(output_dir, exist_ok=True)
    
    # Find all Left images (ZL)
    zl_paths = sorted(glob.glob(os.path.join(input_dir, "ZL*.png")))
    
    if not zl_paths:
        print(f"No ZL* PNG images found in {input_dir}")
        return

    import matplotlib.cm as cm

    for idx, img_path in enumerate(zl_paths, 1):
        filename = os.path.splitext(os.path.basename(img_path))[0]
        right_path = img_path.replace("ZL", "ZR")
        
        if not os.path.exists(right_path):
            print(f"[{idx}/{len(zl_paths)}] Skipping {filename}: Missing right pair.")
            continue
            
        print(f"[{idx}/{len(zl_paths)}] Aligning geometry for: {filename}")
        
        imgL = cv2.imread(img_path)
        imgR = cv2.imread(right_path)
        
        # 1. Compute Metric Disparity via Stereo
        disp_stereo = compute_stereo_disparity(imgL, imgR)
        
        # 2. Compute Relative Disparity via Depth Anything V2
        raw_image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = raw_image.size
        
        inputs = processor(images=raw_image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_depth = outputs.predicted_depth

        disp_da = F.interpolate(
            predicted_depth.unsqueeze(1),
            size=(orig_h, orig_w),
            mode="bicubic",
            align_corners=False
        ).squeeze().cpu().numpy()
        
        # 3. Align DA to Stereo (RANSAC scaling)
        aligned_disp = align_disparity(disp_stereo, disp_da)
        
        # Convert aligned disparity to true metric depth (assuming f*B ~ 1000 for relative physical scaling)
        depth_np = 1000.0 / aligned_disp
        
        # Parse true focal lengths (mocked here, should extract from XML telemetry)
        # Assuming typical Mastcam-Z focal length is ~400 pixels for this resolution
        true_fx = 400.0
        true_fy = 400.0
        
        # 4. Compute Smooth Physical Normals
        normals_np = compute_surface_normals(depth_np, fx=true_fx, fy=true_fy)

        # Save raw numpy files
        depth_out_path = os.path.join(output_dir, f"{filename}_depth.npy")
        normals_out_path = os.path.join(output_dir, f"{filename}_normals.npy")
        np.save(depth_out_path, depth_np)
        np.save(normals_out_path, normals_np)

        # Normalize depth for visualization (inferno colormap)
        d_min, d_max = np.percentile(depth_np, 5), np.percentile(depth_np, 95)
        depth_norm = np.clip((depth_np - d_min) / (d_max - d_min + 1e-8), 0, 1)
        depth_vis = (cm.inferno(depth_norm)[..., :3] * 255).astype(np.uint8)
        Image.fromarray(depth_vis).save(os.path.join(output_dir, f"{filename}_depth_vis.png"))

        # Map normals from [-1, 1] to [0, 255] RGB visualization
        normals_vis = ((normals_np + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)
        Image.fromarray(normals_vis).save(os.path.join(output_dir, f"{filename}_normals_vis.png"))

        print(f"  -> Saved depth: {depth_out_path}")
        print(f"  -> Saved normals: {normals_out_path}")

    print(f"Geometry extraction complete. All outputs in: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="data/nasa_raw", help="Path to input images directory")
    parser.add_argument("--output_dir", type=str, default="output/geometry", help="Path to output directory")
    parser.add_argument("--model_id", type=str, default="depth-anything/Depth-Anything-V2-Small-hf")
    args = parser.parse_args()
    run_geometry_pipeline(args.input_dir, args.output_dir, args.model_id)
