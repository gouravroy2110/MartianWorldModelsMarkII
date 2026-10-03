"""
estimate_geometry.py — Robust Stereoscopic-Aligned Geometry Extraction.
Uses Depth Anything V2 for dense relative geometry, aligned to physical metric scale using 
CAHVOR telemetry-guided stereo matches from ZLF/ZRF image pairs.
"""

import os
import glob
import argparse
import numpy as np
import cv2
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from sklearn.linear_model import RANSACRegressor
import xml.etree.ElementTree as ET
import matplotlib.cm as cm

# Set random seeds for deterministic reproducibility (Agent Rule 4)
np.random.seed(42)
torch.manual_seed(42)

def find_matching_right_pair(left_path):
    """Finds matching right eye image path, handling sub-second timestamp variations."""
    direct = left_path.replace("ZL", "ZR")
    if os.path.exists(direct):
        return direct
    d = os.path.dirname(left_path)
    base = os.path.basename(left_path)
    parts = base.split("_")
    if len(parts) >= 5:
        sol = parts[1]
        seq = parts[4]
        cands = sorted(glob.glob(os.path.join(d, f"ZR*_{sol}_*_{seq}*.png")))
        if cands:
            return cands[0]
    return None

def find_matching_xml(img_path):
    """Finds matching PDS4 XML telemetry file."""
    c1 = img_path.replace("images", "metadata").replace(".png", ".xml")
    if os.path.exists(c1):
        return c1
    c2 = img_path.replace(".png", ".xml")
    if os.path.exists(c2):
        return c2
    d = os.path.dirname(c1)
    base = os.path.basename(img_path)
    parts = base.split("_")
    if len(parts) >= 5:
        cam_prefix = parts[0][:2]
        sol = parts[1]
        seq = parts[4]
        cands = sorted(glob.glob(os.path.join(d, f"{cam_prefix}*_{sol}_*_{seq}*.xml")))
        if cands:
            return cands[0]
    return None

def get_cahv_vector(cahvor_node, name, is_unit=False):
    """Extracts 3D vector from CAHVOR/CAHV XML node."""
    nodes = [c for c in cahvor_node if c.tag.split("}")[-1] == name]
    if not nodes:
        return None
    node = nodes[0]
    vals = []
    suffix = "unit" if is_unit else ("position" if "Center" in name else "pixel")
    for axis in ["x", "y", "z"]:
        child = node.find(f"{{*}}{axis}_{suffix}")
        if child is None or not child.text:
            return None
        vals.append(float(child.text.strip()))
    return np.array(vals, dtype=np.float64)

def parse_camera_telemetry(left_xml_path, right_xml_path, orig_w, orig_h):
    """
    Parses CAHVOR camera telemetry from Left and Right PDS4 XML files.
    Calculates:
      - Physical baseline B = ||C_L - C_R|| (meters)
      - Vergence angle between optical axes A_L and A_R
      - Analytical infinite epipolar disparity (dx_inf, dy_inf)
      - Camera intrinsics fx, fy, cx, cy
    """
    tree_L = ET.parse(left_xml_path)
    root_L = tree_L.getroot()
    tree_R = ET.parse(right_xml_path)
    root_R = tree_R.getroot()

    # Extract sensor dimensions
    raw_samples, raw_lines = None, None
    for elem in root_L.iter():
        tag = elem.tag.split("}")[-1].lower()
        val = (elem.text or "").strip()
        if not val: continue
        if tag == "samples":
            try: raw_samples = float(val)
            except ValueError: pass
        elif tag == "lines":
            try: raw_lines = float(val)
            except ValueError: pass

    if raw_samples is None or raw_lines is None:
        raise ValueError(f"CRITICAL: Missing sensor dimensions ('samples', 'lines') in {left_xml_path}. Halting pipeline.")

    cahvor_L_nodes = [el for el in root_L.iter() if el.tag.split("}")[-1] in ("CAHVOR_Model", "CAHV_Model")]
    cahvor_R_nodes = [el for el in root_R.iter() if el.tag.split("}")[-1] in ("CAHVOR_Model", "CAHV_Model")]

    if not cahvor_L_nodes or not cahvor_R_nodes:
        raise ValueError(f"CRITICAL: CAHVOR camera models missing in {left_xml_path} or {right_xml_path}. Halting pipeline.")

    cahvL = cahvor_L_nodes[0]
    cahvR = cahvor_R_nodes[0]

    CL = get_cahv_vector(cahvL, "Vector_Center", is_unit=False)
    AL = get_cahv_vector(cahvL, "Vector_Axis", is_unit=True)
    HL = get_cahv_vector(cahvL, "Vector_Horizontal", is_unit=False)
    VL = get_cahv_vector(cahvL, "Vector_Vertical", is_unit=False)

    CR = get_cahv_vector(cahvR, "Vector_Center", is_unit=False)
    AR = get_cahv_vector(cahvR, "Vector_Axis", is_unit=True)
    HR = get_cahv_vector(cahvR, "Vector_Horizontal", is_unit=False)
    VR = get_cahv_vector(cahvR, "Vector_Vertical", is_unit=False)

    if any(v is None for v in [CL, AL, HL, VL, CR, AR, HR, VR]):
        raise ValueError(f"CRITICAL: Incomplete CAHVOR vector set in telemetry files. Halting pipeline.")

    # 1. Baseline
    baseline = float(np.linalg.norm(CL - CR))
    if baseline <= 0.01:
        raise ValueError(f"CRITICAL: Stereoscopic baseline is physically invalid ({baseline:.4f} m). Halting pipeline.")

    # 2. Vergence angle
    cos_vergence = np.dot(AL, AR) / (np.linalg.norm(AL) * np.linalg.norm(AR))
    vergence_deg = float(np.degrees(np.arccos(np.clip(cos_vergence, -1.0, 1.0))))
    if vergence_deg > 5.0:
        raise ValueError(f"NON_STEREO: Vergence angle between Left and Right optical axes is {vergence_deg:.2f}° (> 5.0°). Rover mast panned between captures; not a synchronous stereo pair.")

    # 3. Analytical infinite disparity (epipolar rotational vergence shift)
    uL_inf = np.dot(AL, HL) / np.dot(AL, AL)
    vL_inf = np.dot(AL, VL) / np.dot(AL, AL)
    uR_inf = np.dot(AL, HR) / np.dot(AL, AR)
    vR_inf = np.dot(AL, VR) / np.dot(AL, AR)
    dx_inf = float(uL_inf - uR_inf)
    dy_inf = float(vL_inf - vR_inf)

    # 4. Camera Intrinsics
    cx = float(np.dot(HL, AL))
    cy = float(np.dot(VL, AL))
    fx = float(np.linalg.norm(HL - cx * AL))
    fy = float(np.linalg.norm(VL - cy * AL))

    # Scale to current image resolution
    scale_x = orig_w / raw_samples
    scale_y = orig_h / raw_lines
    fx *= scale_x
    fy *= scale_y
    cx *= scale_x
    cy *= scale_y
    dx_inf *= scale_x
    dy_inf *= scale_y

    return {
        "fx": fx, "fy": fy, "cx": cx, "cy": cy,
        "baseline": baseline,
        "dx_inf": dx_inf, "dy_inf": dy_inf,
        "vergence_deg": vergence_deg
    }

def compute_sparse_stereo_disparity(imgL, imgR, dx_inf, dy_inf):
    """
    Extracts physically verified stereo disparity samples using SIFT feature matches
    filtered by epipolar geometry and CAHVOR camera telemetry.
    """
    sift = cv2.SIFT_create(2000)
    kp1, d1 = sift.detectAndCompute(imgL, None)
    kp2, d2 = sift.detectAndCompute(imgR, None)
    if d1 is None or d2 is None or len(d1) < 10 or len(d2) < 10:
        raise ValueError("CRITICAL: Insufficient feature points detected in stereo pair. Halting pipeline.")

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(d1, d2)
    if len(matches) < 10:
        raise ValueError("CRITICAL: Insufficient feature matches between stereo pair. Halting pipeline.")

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    _, inliers = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 3.0, 0.99)
    if inliers is None or np.sum(inliers) < 10:
        raise ValueError("CRITICAL: Fundamental matrix RANSAC failed to find valid epipolar inliers. Halting pipeline.")

    inliers = inliers.ravel() == 1
    pts1_in = pts1[inliers]
    pts2_in = pts2[inliers]

    # Telemetry-guided physical disparity: d = (x_L - x_R) - dx_inf
    disp_true = (pts1_in[:, 0] - pts2_in[:, 0]) - dx_inf
    v_res = np.abs((pts1_in[:, 1] - pts2_in[:, 1]) - dy_inf)

    # Valid epipolar points: vertical residual < 5.0 px and physical disparity > 0
    valid_mask = (v_res < 5.0) & (disp_true > 0)
    valid_pts = pts1_in[valid_mask]
    valid_disp = disp_true[valid_mask]

    if len(valid_disp) < 10:
        raise ValueError(f"CRITICAL: Found only {len(valid_disp)} epipolar-consistent stereo matches. Halting pipeline.")

    return valid_pts, valid_disp

def align_disparity(pts_stereo, disp_stereo, disp_da):
    """
    Aligns Depth Anything relative inverse depth (disp_da) to physical stereo disparity (disp_stereo).
    Fits: disp_stereo = scale * disp_da + shift using robust RANSAC.
    """
    H, W = disp_da.shape
    x_coords = np.clip(np.round(pts_stereo[:, 0]).astype(int), 0, W - 1)
    y_coords = np.clip(np.round(pts_stereo[:, 1]).astype(int), 0, H - 1)
    da_samples = disp_da[y_coords, x_coords]

    ransac = RANSACRegressor(random_state=42)
    ransac.fit(da_samples.reshape(-1, 1), disp_stereo)

    scale = ransac.estimator_.coef_[0]
    shift = ransac.estimator_.intercept_
    print(f"    RANSAC alignment ({len(disp_stereo)} stereo matches): scale={scale:.4f}, shift={shift:.4f}")

    if scale <= 0:
        raise ValueError(f"CRITICAL: RANSAC aligned scale is {scale:.4f} (Negative!). Inverted depth. Halting pipeline.")

    aligned_disp = scale * disp_da + shift
    return np.clip(aligned_disp, 0.1, None)

def compute_surface_normals(depth, fx, fy, cx, cy):
    """Computes smooth normals using bilateral-filtered depth and unprojected 3D point cloud gradients."""
    valid_depth = depth[depth > 0]
    if len(valid_depth) == 0:
        return np.zeros((*depth.shape, 3), dtype=np.float32)

    depth_med = np.median(valid_depth)
    mad = np.median(np.abs(valid_depth - depth_med))
    dynamic_sigma_color = max(0.1, mad * 0.05)

    depth_smooth = cv2.bilateralFilter(depth.astype(np.float32), d=9, sigmaColor=dynamic_sigma_color, sigmaSpace=75)

    H, W = depth.shape
    y_coords, x_coords = np.indices((H, W))

    Z = depth_smooth.astype(np.float32)
    X = ((x_coords - cx) * Z / fx).astype(np.float32)
    Y = ((y_coords - cy) * Z / fy).astype(np.float32)

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

    # Ensure normals face camera (-Z in standard OpenCV coords)
    flip_mask = normals[:, :, 2] > 0
    normals[flip_mask] = -normals[flip_mask]

    return normals

def run_geometry_pipeline(input_dir: str, output_dir: str, model_id: str = "depth-anything/Depth-Anything-V2-Small-hf"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading Depth Anything V2 ({model_id}) on {device}...")

    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device)
    model.eval()

    os.makedirs(output_dir, exist_ok=True)

    zl_paths = sorted(glob.glob(os.path.join(input_dir, "ZL*.png")))
    if not zl_paths:
        print(f"No ZL* PNG images found in {input_dir}")
        return

    processed_count = 0
    for idx, img_path in enumerate(zl_paths, 1):
        filename = os.path.splitext(os.path.basename(img_path))[0]
        right_path = find_matching_right_pair(img_path)

        if not right_path or not os.path.exists(right_path):
            print(f"[{idx}/{len(zl_paths)}] Skipping {filename}: Missing right pair.")
            continue

        xml_left = find_matching_xml(img_path)
        xml_right = find_matching_xml(right_path)

        if not xml_left or not os.path.exists(xml_left):
            raise FileNotFoundError(f"CRITICAL: Missing required left XML telemetry for: {img_path}")
        if not xml_right or not os.path.exists(xml_right):
            raise FileNotFoundError(f"CRITICAL: Missing required right XML telemetry for: {right_path}")

        raw_image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = raw_image.size

        # 1. Parse Camera Telemetry & Intrinsics
        try:
            telem = parse_camera_telemetry(xml_left, xml_right, orig_w, orig_h)
        except ValueError as e:
            if "NON_STEREO" in str(e):
                print(f"[{idx}/{len(zl_paths)}] Skipping {filename}: {e}")
                continue
            raise e

        print(f"[{idx}/{len(zl_paths)}] Aligning geometry for: {filename}")
        print(f"  -> Extracted Telemetry: fx={telem['fx']:.1f} px, fy={telem['fy']:.1f} px, Baseline B={telem['baseline']:.4f} m, fx*B={telem['fx']*telem['baseline']:.1f} px*m, vergence={telem['vergence_deg']:.2f}°")

        imgL = cv2.imread(img_path)
        imgR = cv2.imread(right_path)

        # 2. Extract physically verified stereo matches guided by CAHVOR epipolar geometry
        pts_stereo, disp_stereo = compute_sparse_stereo_disparity(imgL, imgR, telem["dx_inf"], telem["dy_inf"])

        # 3. Compute Relative Inverse Depth via Depth Anything V2
        inputs = processor(images=raw_image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_depth_estimation(outputs, target_sizes=[(orig_h, orig_w)])
        predicted_depth = results[0]["predicted_depth"]
        disp_da = predicted_depth.squeeze().cpu().numpy()

        # 4. Align Depth Anything to Physical Stereo (RANSAC scaling)
        aligned_disp = align_disparity(pts_stereo, disp_stereo, disp_da)

        # 5. Convert aligned disparity to true metric depth in SI units (meters)
        depth_np = (telem["fx"] * telem["baseline"]) / aligned_disp

        # 6. Compute Smooth Physical Normals
        normals_np = compute_surface_normals(depth_np, fx=telem["fx"], fy=telem["fy"], cx=telem["cx"], cy=telem["cy"])

        # Save raw numpy files
        depth_out_path = os.path.join(output_dir, f"{filename}_depth.npy")
        normals_out_path = os.path.join(output_dir, f"{filename}_normals.npy")
        np.save(depth_out_path, depth_np)
        np.save(normals_out_path, normals_np)

        # Normalize depth for visualization (inferno colormap)
        d_min, d_max = np.percentile(depth_np, 5), np.percentile(depth_np, 95)
        depth_norm = np.clip((depth_np - d_min) / (d_max - d_min + 1e-8), 0, 1)
        depth_vis = (cm.inferno(depth_norm)[..., :3] * 255).astype(np.uint8)

        # Map normals from [-1, 1] to [0, 255] RGB visualization
        normals_vis = ((normals_np + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)

        # Generate single combined diagnostic sprite (RGB | Depth | Normals)
        orig_img_resized = cv2.resize(imgL, (orig_w, orig_h))
        orig_img_rgb = cv2.cvtColor(orig_img_resized, cv2.COLOR_BGR2RGB)

        diagnostic_sprite = np.hstack((orig_img_rgb, depth_vis[:, :, :3], normals_vis))
        diag_path = os.path.join(output_dir, f"{filename}_diagnostic.png")
        Image.fromarray(diagnostic_sprite).save(diag_path)

        print(f"  -> Saved diagnostic combined sprite: {diag_path}")
        processed_count += 1

    print(f"\n================== GEOMETRY PIPELINE SUCCESS ==================")
    print(f"Successfully processed and validated {processed_count} synchronous stereo pairs.")
    print(f"All physical outputs saved in: {output_dir}")
    print("===============================================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="data/nasa_raw", help="Path to input images directory")
    parser.add_argument("--output_dir", type=str, default="output/geometry", help="Path to output directory")
    parser.add_argument("--model_id", type=str, default="depth-anything/Depth-Anything-V2-Small-hf")
    args = parser.parse_args()
    run_geometry_pipeline(args.input_dir, args.output_dir, args.model_id)
