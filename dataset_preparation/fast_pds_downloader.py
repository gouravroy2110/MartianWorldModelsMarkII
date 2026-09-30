#!/usr/bin/env python3
import os
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

def main():
    input_file = sys.argv[1] if len(sys.argv) > 1 else "manifest.txt"
    dest_dir = sys.argv[2] if len(sys.argv) > 2 else "/tmp/martian_staging"
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    
    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        sys.exit(1)
        
    with open(input_file, "r") as f:
        lines = f.readlines()
        
    tasks = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
            
        # Case A: Plain URL manifest (e.g. site1_manifest.txt)
        if line.startswith("http://") or line.startswith("https://"):
            url = line
            filename = os.path.basename(url.split("?")[0])
            target_path = os.path.join(dest_dir, filename)
            tasks.append((url, target_path))
            continue
            
        # Case B: Wget script line (e.g. ... -O <filepath> <url>)
        m = re.search(r"-O\s+(\S+)\s+(\S+)", line)
        if m:
            target_path = m.group(1)
            raw_url = m.group(2)
            
            # Map release ID num
            rel_m = re.search(r"::(\d+)$", raw_url)
            rel_id = int(rel_m.group(1)) if rel_m else 0
            rel_folder = "cumulative" if rel_id == 0 else f"r{rel_id}"
            
            # Map JPL API endpoint to CloudFront direct CDN (bypassing AWS WAF 403)
            url = raw_url.replace(
                "https://pds-imaging.jpl.nasa.gov/api/data/atlas:pds4:mars_2020:perseverance:/",
                f"https://d1ejlg980osaur.cloudfront.net/m20/{rel_folder}/"
            )
            url = re.sub(r"::\d+$", "", url)
            
            if dest_dir:
                filename = os.path.basename(target_path)
                target_path = os.path.join(dest_dir, filename)
                
            tasks.append((url, target_path))
            
    # Deduplicate tasks by destination file
    tasks_dict = {path: (u, path) for u, path in tasks}
    tasks = list(tasks_dict.values())
    
    print(f"Extracted {len(tasks)} unique files to download into {dest_dir}.")
    if not tasks:
        return

    os.makedirs(dest_dir, exist_ok=True)

    def download_item(item):
        u, p = item
        if os.path.exists(p) and os.path.getsize(p) > 0:
            return "skipped"
        try:
            req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp, open(p, "wb") as out:
                out.write(resp.read())
            return "ok"
        except Exception:
            # Retry once on failure
            try:
                time.sleep(1)
                req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=30) as resp, open(p, "wb") as out:
                    out.write(resp.read())
                return "ok"
            except Exception as e:
                return f"failed: {e}"

    success = 0
    skipped = 0
    failed = 0
    t0 = time.time()
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(download_item, t): t for t in tasks}
        total = len(tasks)
        for i, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            if res == "ok":
                success += 1
            elif res == "skipped":
                skipped += 1
            else:
                failed += 1
                
            if i % 100 == 0 or i == total:
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 0
                print(f"[{i}/{total}] {success} ok, {skipped} skipped, {failed} failed ({rate:.1f} files/s, {elapsed:.1f}s)")
                
    total_time = time.time() - t0
    print(f"\nCompleted in {total_time:.1f}s: {success} downloaded, {skipped} skipped, {failed} failed.")

if __name__ == "__main__":
    main()
