"""
visualize_summary.py — Stitches diagnostic verification panels for each image:
[ Original Image | Predicted Depth Map | Surface Normals | Sun Vector Overlay ]
"""

import os
import glob
import argparse
from PIL import Image, ImageDraw, ImageFont


def build_summary_grid(image_dir: str, geometry_dir: str, sun_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))

    for img_path in image_paths:
        filename = os.path.splitext(os.path.basename(img_path))[0]
        depth_vis_path = os.path.join(geometry_dir, f"{filename}_depth_vis.png")
        normals_vis_path = os.path.join(geometry_dir, f"{filename}_normals_vis.png")
        sun_overlay_path = os.path.join(sun_dir, f"{filename}_sun_overlay.png")

        if not (os.path.exists(depth_vis_path) and os.path.exists(normals_vis_path) and os.path.exists(sun_overlay_path)):
            continue

        im_orig = Image.open(img_path).convert("RGB")
        im_depth = Image.open(depth_vis_path).convert("RGB")
        im_normals = Image.open(normals_vis_path).convert("RGB")
        im_sun = Image.open(sun_overlay_path).convert("RGB")

        # Resize all panels to uniform display size (e.g., width 640)
        panel_w = 640
        aspect = panel_w / im_orig.width
        panel_h = int(im_orig.height * aspect)

        panels = [
            ("Original Mastcam-Z RGB", im_orig.resize((panel_w, panel_h), Image.Resampling.BILINEAR)),
            ("Depth Anything V2 Depth", im_depth.resize((panel_w, panel_h), Image.Resampling.NEAREST)),
            ("Surface Normals (Camera Space)", im_normals.resize((panel_w, panel_h), Image.Resampling.NEAREST)),
            ("Sun Vector Compass Overlay", im_sun.resize((panel_w, panel_h), Image.Resampling.BILINEAR))
        ]

        # 2x2 grid layout
        grid_w = panel_w * 2 + 30
        grid_h = panel_h * 2 + 70
        grid = Image.new("RGB", (grid_w, grid_h), color=(25, 25, 25))
        draw = ImageDraw.Draw(grid)

        positions = [
            (10, 30),
            (panel_w + 20, 30),
            (10, panel_h + 50),
            (panel_w + 20, panel_h + 50)
        ]

        for (title, panel_img), (x, y) in zip(panels, positions):
            grid.paste(panel_img, (x, y))
            draw.text((x + 10, y - 20), title, fill=(240, 240, 240))

        out_grid_path = os.path.join(output_dir, f"{filename}_diagnostic_grid.jpg")
        grid.save(out_grid_path, quality=92)
        print(f"Saved diagnostic grid: {out_grid_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, default="data/nasa_raw")
    parser.add_argument("--geometry_dir", type=str, default="output/geometry")
    parser.add_argument("--sun_dir", type=str, default="output/sun_estimation")
    parser.add_argument("--output_dir", type=str, default="output/diagnostic_grids")
    args = parser.parse_args()

    build_summary_grid(args.image_dir, args.geometry_dir, args.sun_dir, args.output_dir)
