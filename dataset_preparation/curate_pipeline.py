#!/usr/bin/env python3
"""
Automated Mars Rover Image Quality & Filtering Pipeline (Gates 1 to 3)

Filters out:
  Gate 1: Calibration targets (steep downward pitch) and direct Sun shots (Filter 7 / steep upward pitch)
  Gate 2: Blurry images (low Laplacian variance) and severe exposure clipping
  Gate 3: Rover chassis / deck intrusions (boundary inspection and rover body masking)
"""

import os
import re
import cv2
import json
import shutil
import sqlite3
import argparse
import numpy as np
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

def parse_xml_telemetry(xml_path):
    """Extracts pointing, solar, and instrument parameters from PDS4 XML label."""
    info = {
        "sol": None,
        "site_id": None,
        "drive_id": None,
        "mast_elevation": None,
        "mast_azimuth": None,
        "solar_elevation": None,
        "solar_azimuth": None,
        "filter_name": "",
        "target_name": "Mars",
        "ltst": ""
    }
    if not os.path.exists(xml_path):
        return info

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for elem in root.iter():
            tag = elem.tag.split("}")[-1].lower()
            val = (elem.text or "").strip()
            if not val:
                continue

            if tag == "mast_elevation" and info["mast_elevation"] is None:
                try: info["mast_elevation"] = float(val)
                except ValueError: pass
            elif tag == "mast_azimuth" and info["mast_azimuth"] is None:
                try: info["mast_azimuth"] = float(val)
                except ValueError: pass
            elif tag == "solar_elevation" and info["solar_elevation"] is None:
                try: info["solar_elevation"] = float(val)
                except ValueError: pass
            elif tag == "solar_azimuth" and info["solar_azimuth"] is None:
                try: info["solar_azimuth"] = float(val)
                except ValueError: pass
            elif tag == "site_id" and info["site_id"] is None:
                try: info["site_id"] = int(val)
                except ValueError: pass
            elif tag == "drive_id" and info["drive_id"] is None:
                try: info["drive_id"] = int(val)
                except ValueError: pass
            elif tag == "target_name":
                info["target_name"] = val
            elif "filter_name" in tag and not info["filter_name"]:
                info["filter_name"] = val
            elif tag in ["local_true_solar_time", "local_mean_solar_time"] and not info["ltst"]:
                info["ltst"] = val
    except Exception:
        pass

    return info

def gate1_telemetry_check(fname, xml_info):
    """
    Gate 1: Telemetry & Pointing Check
    Rejects:
      - Calibration target shots on the rover deck (mast pointing steeply downward into deck)
      - Direct Sun observation / atmospheric optical depth (Filter 7 / pointing high up into sky)
    """
    # 1. Reject by explicit filter name or filename code
    if "ZR7" in fname or "ZL7" in fname or "_113" in fname:
        return False, "Gate 1: Direct Sun / Solar Tau filter (Filter 7)"
    
    # 2. Check target name
    target = xml_info.get("target_name", "").lower()
    if "calibration" in target or "cal target" in target:
        return False, "Gate 1: Target labeled as Calibration Target"

    # 3. Check mast elevation angle
    mast_el = xml_info.get("mast_elevation")
    if mast_el is not None:
        # Rover deck is directly below/behind mast (typically -40 deg to -75 deg)
        if mast_el < -35.0:
            return False, f"Gate 1: Calibration Target / Rover Deck (mast_elevation={mast_el:.1f}° < -35°)"
        # Direct sky shots looking straight up (typically > +25 deg unless pointing at distant hills)
        if mast_el > +25.0:
            return False, f"Gate 1: Direct Sky / Solar Observation (mast_elevation={mast_el:.1f}° > +25°)"

    return True, "Passed Gate 1"

def gate2_quality_check(img_path, min_sharpness=50.0):
    """
    Gate 2: Image Quality & Exposure Check
    Rejects:
      - Blurred images (wind vibration, camera motion, dust clouds)
      - Severe over/under-exposure (clipped histograms)
    """
    img = cv2.imread(img_path)
    if img is None:
        return False, 0.0, "Gate 2: Could not decode image file"

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Laplacian sharpness metric
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < min_sharpness:
        return False, laplacian_var, f"Gate 2: Blurred frame (sharpness={laplacian_var:.1f} < {min_sharpness})"

    # 2. Exposure clipping (overexposure / saturated highlights)
    h, w = gray.shape
    total_pixels = h * w
    saturated_pixels = np.sum(gray >= 254)
    black_pixels = np.sum(gray <= 2)

    if (saturated_pixels / total_pixels) > 0.20:
        return False, laplacian_var, "Gate 2: Severe overexposure (>20% pixels saturated white)"
    if (black_pixels / total_pixels) > 0.40:
        return False, laplacian_var, "Gate 2: Severe underexposure (>40% pixels black)"

    return True, laplacian_var, f"Passed Gate 2 (sharpness={laplacian_var:.1f})"

def gate3_rover_body_check(img_path, xml_info):
    """
    Gate 3: Rover Body / Chassis Intrusion Check
    When Mastcam-Z points forward-down at rocks (elevation between -25° and -35°),
    the bottom corners may capture parts of the rover wheels or robotic arm.
    """
    mast_el = xml_info.get("mast_elevation")
    # If mast elevation is level or up, rover body cannot physically be in frame
    if mast_el is not None and mast_el >= -15.0:
        return True, "Passed Gate 3: Elevation clear of chassis"

    # Analyze bottom 15% of the frame for rover body characteristics (metallic gray, dark high-contrast edges)
    img = cv2.imread(img_path)
    if img is None:
        return True, "Passed Gate 3 (no image)"

    h, w, _ = img.shape
    bottom_crop = img[int(h * 0.85):, :]
    
    # Mars terrain has high Red/Blue ratio (> 1.4). Rover painted/metallic parts have neutral R/B ratio (~ 1.0)
    b, g, r = cv2.split(bottom_crop)
    r_float = r.astype(np.float32) + 1e-5
    b_float = b.astype(np.float32) + 1e-5
    rb_ratio = r_float / b_float
    neutral_pixels = np.sum(rb_ratio < 1.15)
    total_bottom_pixels = bottom_crop.shape[0] * bottom_crop.shape[1]

    if (neutral_pixels / total_bottom_pixels) > 0.35:
        return False, "Gate 3: Significant rover chassis/wheel intrusion detected in lower frame"

    return True, "Passed Gate 3"

def filter_dataset(staging_dir, output_dir, min_sharpness=50.0, copy_files=False):
    """Executes Gates 1 to 3 across all staged images."""
    os.makedirs(output_dir, exist_ok=True)
    
    png_files = [f for f in os.listdir(staging_dir) if f.endswith(".png")]
    print(f"[*] Starting Curation Filter across {len(png_files)} staged images in {staging_dir}...")

    stats = {
        "total": len(png_files),
        "passed": 0,
        "dropped_gate1": 0,
        "dropped_gate2": 0,
        "dropped_gate3": 0
    }

    curated_records = []

    for idx, fname in enumerate(sorted(png_files), 1):
        png_path = os.path.join(staging_dir, fname)
        xml_path = os.path.splitext(png_path)[0] + ".xml"
        xml_info = parse_xml_telemetry(xml_path)

        # Gate 1: Telemetry (Cal target & Sun)
        g1_pass, g1_reason = gate1_telemetry_check(fname, xml_info)
        if not g1_pass:
            stats["dropped_gate1"] += 1
            continue

        # Gate 2: Quality & Sharpness
        g2_pass, sharpness, g2_reason = gate2_quality_check(png_path, min_sharpness)
        if not g2_pass:
            stats["dropped_gate2"] += 1
            continue

        # Gate 3: Rover Body Intrusion
        g3_pass, g3_reason = gate3_rover_body_check(png_path, xml_info)
        if not g3_pass:
            stats["dropped_gate3"] += 1
            continue

        # All 3 Gates Passed!
        stats["passed"] += 1
        record = {
            "filename": fname,
            "sol": xml_info.get("sol"),
            "site_id": xml_info.get("site_id"),
            "drive_id": xml_info.get("drive_id"),
            "mast_elevation": xml_info.get("mast_elevation"),
            "solar_elevation": xml_info.get("solar_elevation"),
            "solar_azimuth": xml_info.get("solar_azimuth"),
            "sharpness": round(sharpness, 1),
            "ltst": xml_info.get("ltst")
        }
        curated_records.append(record)

        if copy_files:
            # Copy PNG, XML, and IMG (if exists)
            base_no_ext = os.path.splitext(png_path)[0]
            for ext in [".png", ".xml", ".IMG"]:
                src = base_no_ext + ext
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(output_dir, os.path.basename(src)))

        if idx % 100 == 0 or idx == len(png_files):
            print(f"    Processed {idx}/{len(png_files)}: {stats['passed']} passed, {stats['dropped_gate1']} dropped (G1), {stats['dropped_gate2']} dropped (G2), {stats['dropped_gate3']} dropped (G3)")

    # Save curation index
    manifest_out = os.path.join(output_dir, "curated_manifest.json")
    with open(manifest_out, "w") as f:
        json.dump(curated_records, f, indent=2)

    print("\n================== CURATION SUMMARY ==================")
    print(f"Total Staged Images Evaluated: {stats['total']}")
    print(f"Passed All Gates (High-Quality Martian Terrain): {stats['passed']} ({stats['passed']/max(1, stats['total'])*100:.1f}%)")
    print(f"Dropped by Gate 1 (Cal Target / Sun):           {stats['dropped_gate1']}")
    print(f"Dropped by Gate 2 (Blur / Poor Exposure):       {stats['dropped_gate2']}")
    print(f"Dropped by Gate 3 (Rover Body Intrusion):       {stats['dropped_gate3']}")
    print(f"Saved manifest to: {manifest_out}")
    print("======================================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Curate Mars Rover Dataset (Gates 1-3)")
    parser.add_argument("--staging", type=str, default="/tmp/martian_staging", help="Input staging directory")
    parser.add_argument("--output", type=str, default="/kaggle/working/curated_martian_dataset", help="Output curated directory")
    parser.add_argument("--min-sharpness", type=float, default=60.0, help="Minimum Laplacian sharpness score")
    parser.add_argument("--copy", action="store_true", help="Copy passed files to output directory")
    args = parser.parse_args()

    filter_dataset(args.staging, args.output, min_sharpness=args.min_sharpness, copy_files=args.copy)
