"""
Download Mars 2020 Mastcam-Z raw/calibrated images and XML metadata from NASA PDS CloudFront CDN.
Supports local download and Kaggle remote download.
"""

import os
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

RAW_URLS = [
    "https://pds-imaging.jpl.nasa.gov/api/data/atlas:pds4:mars_2020:perseverance:/mars2020_mastcamz_ops_calibrated/data/sol/01856/ids/rdr/zcam/ZLF_1856_0831715247_803RZS_N0880620ZCAM07125_1100LMJ01.IMG::16",
    "https://pds-imaging.jpl.nasa.gov/api/data/atlas:pds4:mars_2020:perseverance:/mars2020_mastcamz_ops_calibrated/data/sol/01856/ids/rdr/zcam/ZLF_1856_0831715247_803RZS_N0880620ZCAM07125_1100LMJ01.xml::16",
    "https://pds-imaging.jpl.nasa.gov/api/data/atlas:pds4:mars_2020:perseverance:/mars2020_mastcamz_ops_calibrated/browse/sol/01856/ids/rdr/zcam/ZLF_1856_0831715247_803RZS_N0880620ZCAM07125_1100LMJ01.png::16",
    "https://pds-imaging.jpl.nasa.gov/api/data/atlas:pds4:mars_2020:perseverance:/mars2020_mastcamz_ops_calibrated/data/sol/01856/ids/rdr/zcam/ZRF_1856_0831715247_803RZS_N0880620ZCAM07125_1100LMJ01.IMG::16",
    "https://pds-imaging.jpl.nasa.gov/api/data/atlas:pds4:mars_2020:perseverance:/mars2020_mastcamz_ops_calibrated/data/sol/01856/ids/rdr/zcam/ZRF_1856_0831715247_803RZS_N0880620ZCAM07125_1100LMJ01.xml::16",
    "https://pds-imaging.jpl.nasa.gov/api/data/atlas:pds4:mars_2020:perseverance:/mars2020_mastcamz_ops_calibrated/browse/sol/01856/ids/rdr/zcam/ZRF_1856_0831715247_803RZS_N0880620ZCAM07125_1100LMJ01.png::16",
    "https://pds-imaging.jpl.nasa.gov/api/data/atlas:pds4:mars_2020:perseverance:/mars2020_mastcamz_ops_calibrated/data/sol/01855/ids/rdr/zcam/ZRF_1855_0831623098_145RZS_N0880206ZCAM09913_1100LMJ01.IMG::16",
    "https://pds-imaging.jpl.nasa.gov/api/data/atlas:pds4:mars_2020:perseverance:/mars2020_mastcamz_ops_calibrated/data/sol/01855/ids/rdr/zcam/ZRF_1855_0831623098_145RZS_N0880206ZCAM09913_1100LMJ01.xml::16",
    "https://pds-imaging.jpl.nasa.gov/api/data/atlas:pds4:mars_2020:perseverance:/mars2020_mastcamz_ops_calibrated/browse/sol/01855/ids/rdr/zcam/ZRF_1855_0831623098_145RZS_N0880206ZCAM09913_1100LMJ01.png::16",
    "https://pds-imaging.jpl.nasa.gov/api/data/atlas:pds4:mars_2020:perseverance:/mars2020_mastcamz_ops_calibrated/data/sol/01855/ids/rdr/zcam/ZLF_1855_0831623046_148RZS_N0880206ZCAM09913_1100LMJ01.IMG::16",
    "https://pds-imaging.jpl.nasa.gov/api/data/atlas:pds4:mars_2020:perseverance:/mars2020_mastcamz_ops_calibrated/data/sol/01855/ids/rdr/zcam/ZLF_1855_0831623046_148RZS_N0880206ZCAM09913_1100LMJ01.xml::16",
    "https://pds-imaging.jpl.nasa.gov/api/data/atlas:pds4:mars_2020:perseverance:/mars2020_mastcamz_ops_calibrated/browse/sol/01855/ids/rdr/zcam/ZLF_1855_0831623046_148RZS_N0880206ZCAM09913_1100LMJ01.png::16"
]

def map_to_cdn_url(raw_url: str) -> str:
    """Map JPL Atlas PDS4 URL with ::release_id to direct CloudFront CDN mirror."""
    rel_m = re.search(r"::(\d+)$", raw_url)
    rel_id = int(rel_m.group(1)) if rel_m else 16
    rel_folder = f"r{rel_id}" if rel_id > 0 else "cumulative"
    
    clean_url = raw_url.replace(
        "https://pds-imaging.jpl.nasa.gov/api/data/atlas:pds4:mars_2020:perseverance:/",
        f"https://d1ejlg980osaur.cloudfront.net/m20/{rel_folder}/"
    )
    clean_url = re.sub(r"::\d+$", "", clean_url)
    return clean_url

def download_file(url: str, output_path: str, max_retries: int = 3) -> bool:
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        print(f"[EXISTS] {os.path.basename(output_path)} ({os.path.getsize(output_path):,} bytes)")
        return True

    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
                with open(output_path, "wb") as f:
                    f.write(data)
            print(f"[OK] Downloaded {os.path.basename(output_path)} ({len(data):,} bytes)")
            return True
        except Exception as e:
            print(f"[RETRY {attempt}/{max_retries}] Failed {os.path.basename(output_path)}: {e}")
            time.sleep(2)

    print(f"[FAILED] Could not download {url}")
    return False

def main():
    dest_dir = sys.argv[1] if len(sys.argv) > 1 else "data/nasa_raw"
    os.makedirs(dest_dir, exist_ok=True)
    
    tasks = []
    for raw_url in RAW_URLS:
        cdn_url = map_to_cdn_url(raw_url)
        filename = os.path.basename(cdn_url)
        target_path = os.path.join(dest_dir, filename)
        tasks.append((cdn_url, target_path))

    print(f"Downloading {len(tasks)} files to: {os.path.abspath(dest_dir)}")
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(download_file, url, path): (url, path) for url, path in tasks}
        for fut in as_completed(futures):
            fut.result()

    print("Download completed.")

if __name__ == "__main__":
    main()
