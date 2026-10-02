"""
verify_geometry.py — Visual Verification of Depth and Surface Normals.
Generates Novel View (Test A) and Virtual Flashlight (Test B) images to physically verify
the correctness of extracted depth and normal maps.
"""

import os
import glob
import argparse
import numpy as np
import cv2

def render_novel_view(img, depth, shift_x=30):
    H, W, _ = img.shape
    y, x = np.indices((H, W))
    
    shift_magnitude = (depth.max() - depth) / (depth.max() - depth.min() + 1e-5)
    new_x = np.clip(x + shift_x * shift_magnitude, 0, W-1).astype(int)
    
    warped = np.zeros_like(img)
    warped[y, new_x] = img[y, x]
    
    # Morphology to fill 1px holes
    kernel = np.ones((3,3), np.uint8)
    warped = cv2.morphologyEx(warped, cv2.MORPH_CLOSE, kernel)
    return warped

def virtual_flashlight(normals, light_dir):
    intensity = np.clip(np.dot(normals, light_dir), 0, 1)
    return (intensity * 255).astype(np.uint8)

def run_verification(input_dir: str, geometry_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    zl_paths = sorted(glob.glob(os.path.join(input_dir, "ZL*.png")))
    
    if not zl_paths:
        print(f"No ZL* PNG images found in {input_dir}")
        return

    for img_path in zl_paths:
        filename = os.path.splitext(os.path.basename(img_path))[0]
        depth_path = os.path.join(geometry_dir, f"{filename}_depth.npy")
        normals_path = os.path.join(geometry_dir, f"{filename}_normals.npy")
        
        if not os.path.exists(depth_path) or not os.path.exists(normals_path):
            print(f"Skipping {filename}: Geometry files not found.")
            continue
            
        print(f"Generating visual verification for: {filename}")
        
        img = cv2.imread(img_path)
        depth = np.load(depth_path)
        normals = np.load(normals_path)
        
        # Test A: Novel View Parallax
        novel_view = render_novel_view(img, depth, shift_x=50)
        cv2.imwrite(os.path.join(output_dir, f"{filename}_testA_novel_view.png"), novel_view)
        
        # Test B: Virtual Flashlight (Left and Right)
        l1 = np.array([0.8, -0.2, 0.5])
        l1 = l1 / np.linalg.norm(l1)
        l2 = np.array([-0.8, -0.2, 0.5])
        l2 = l2 / np.linalg.norm(l2)
        
        flash1 = virtual_flashlight(normals, l1)
        flash2 = virtual_flashlight(normals, l2)
        
        # Combine side by side
        concat_flash = np.concatenate((flash1, flash2), axis=1)
        cv2.imwrite(os.path.join(output_dir, f"{filename}_testB_flashlight.png"), concat_flash)
        
    print(f"Verification complete. Outputs saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="data/nasa_raw")
    parser.add_argument("--geometry_dir", type=str, default="output/geometry")
    parser.add_argument("--output_dir", type=str, default="output/verification")
    args = parser.parse_args()
    
    run_verification(args.input_dir, args.geometry_dir, args.output_dir)
