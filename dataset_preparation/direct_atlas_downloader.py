#!/usr/bin/env python3
"""
Direct NASA PDS Atlas Crawler & CloudFront Fast Downloader
Queries the Atlas Elasticsearch API directly and streams files at high speed
from NASA's CloudFront CDN directly into /tmp/martian_staging.
"""

import os
import sys
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

SEARCH_API = "https://pds-imaging.jpl.nasa.gov/api/search/atlas/_search"
CLOUDFRONT_BASE = "https://d1ejlg980osaur.cloudfront.net/m20"

SITES = [
    {
        "id": "site1",
        "name": "Landing Site Bedrock (Octavia E. Butler Landing)",
        "sols": "15-30",
        "date_range": "[2021-03-06 TO 2021-03-22]"
    },
    {
        "id": "site2",
        "name": "South Seitah Dunes & Megaripples",
        "sols": "195-225",
        "date_range": "[2021-09-07 TO 2021-10-09]"
    },
    {
        "id": "site3",
        "name": "Delta Front Scarp & Enchanted Lake",
        "sols": "420-445",
        "date_range": "[2022-04-26 TO 2022-05-23]"
    },
    {
        "id": "site4",
        "name": "Skinner Ridge (Delta Top)",
        "sols": "455-475",
        "date_range": "[2022-06-01 TO 2022-06-22]"
    },
    {
        "id": "site5",
        "name": "Tenby Mudstones (Amalik)",
        "sols": "520-545",
        "date_range": "[2022-08-07 TO 2022-09-02]"
    },
    {
        "id": "site6",
        "name": "Margin Carbonate Unit (Jurabi Point)",
        "sols": "955-985",
        "date_range": "[2023-10-28 TO 2023-11-28]"
    },
    {
        "id": "site7",
        "name": "Bright Angel Channel & Neretva Vallis",
        "sols": "1165-1190",
        "date_range": "[2024-05-30 TO 2024-06-26]"
    },
    {
        "id": "site8",
        "name": "Crater Rim Lookout (Aurora Peak)",
        "sols": "1830-1860",
        "date_range": "[2026-04-14 TO 2026-05-16]"
    }
]

def build_query(date_range_str, product_type="RZS"):
    qs = (
        f"_exists_:gather.uri AND "
        f"(gather.common.spacecraft:perseverance AND "
        f"gather.common.instrument:(MCZ_LEFT OR MCZ_RIGHT) AND "
        f"gather.common.kind:regular AND "
        f"gather.pds_archive.bundle_id:mars2020_mastcamz_ops_calibrated AND "
        f"gather.common.product_type:{product_type} AND "
        f"gather.time.start_time:{date_range_str})"
    )
    return qs

def query_atlas_all(date_range_str, product_type="RZS", max_results=5000):
    """Paginates through Atlas Elasticsearch API to retrieve all matching items."""
    qs = build_query(date_range_str, product_type)
    offset = 0
    page_size = 100
    all_hits = []

    print(f"[*] Querying Atlas API for range: {date_range_str} ({product_type})...")
    while True:
        payload = {
            "query": {"query_string": {"query": qs}},
            "from": offset,
            "size": page_size,
            "sort": [{"gather.time.start_time": "asc"}, {"uri": "asc"}],
            "_source": [
                "uri", "gather.uri", "gather.pds_archive", "release_id_num",
                "archive.size", "gather.time.start_time"
            ]
        }
        
        req = urllib.request.Request(
            SEARCH_API,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"}
        )
        
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            print(f"[!] API query error at offset {offset}: {e}")
            break

        hits = data.get("hits", {}).get("hits", [])
        total = data.get("hits", {}).get("total", {}).get("value", 0)
        
        if not hits:
            break
            
        all_hits.extend(hits)
        print(f"    Fetched {len(all_hits)} / {total} entries...")
        offset += len(hits)
        
        if offset >= total or len(all_hits) >= max_results:
            break
            
        time.sleep(0.2)  # courteous API rate limit
        
    print(f"[+] Total retrieved: {len(all_hits)} records.")
    return all_hits

def hit_to_download_urls(hit):
    """
    Extracts CloudFront download URLs for:
    1. Calibrated Image (.IMG)
    2. PDS4 Metadata Label (.xml)
    3. Browse Preview (.png)
    """
    source = hit.get("_source", {})
    uri = source.get("uri") or source.get("gather", {}).get("uri", "")
    release_id = source.get("release_id_num", 0)
    
    # Determine CloudFront subfolder
    rel_folder = "cumulative" if (release_id == 0 or release_id is None) else f"r{release_id}"
    
    # Extract related files from pds_archive
    pds_archive = source.get("gather", {}).get("pds_archive", {})
    related = pds_archive.get("related", {})
    
    downloads = []
    
    def to_cf_url(raw_uri):
        prefix = "atlas:pds4:mars_2020:perseverance:/"
        if raw_uri.startswith(prefix):
            rel_path = raw_uri[len(prefix):]
            return f"{CLOUDFRONT_BASE}/{rel_folder}/{rel_path}"
        return None

    # Main image (.IMG)
    if uri:
        cf = to_cf_url(uri)
        if cf:
            fname = os.path.basename(cf)
            downloads.append((cf, fname))
            
    # XML label (.xml)
    label_uri = related.get("label", {}).get("uri")
    if label_uri:
        cf = to_cf_url(label_uri)
        if cf:
            fname = os.path.basename(cf)
            downloads.append((cf, fname))
            
    # Browse preview (.png)
    browse_uri = related.get("browse", {}).get("uri")
    if browse_uri:
        cf = to_cf_url(browse_uri)
        if cf:
            fname = os.path.basename(cf)
            downloads.append((cf, fname))
            
    return downloads

def download_file(item, dest_dir):
    url, filename = item
    target_path = os.path.join(dest_dir, filename)
    
    if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
        return "skipped"
        
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp, open(target_path, "wb") as out:
            out.write(resp.read())
        return "ok"
    except Exception as e:
        # Retry once
        try:
            time.sleep(1)
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp, open(target_path, "wb") as out:
                out.write(resp.read())
            return "ok"
        except Exception:
            return f"failed: {e}"

def crawl_and_download(date_range_str, dest_dir, product_type="RZS", max_workers=16, include_img=True, include_xml=True, include_png=True):
    os.makedirs(dest_dir, exist_ok=True)
    hits = query_atlas_all(date_range_str, product_type)
    
    all_items = []
    for hit in hits:
        urls = hit_to_download_urls(hit)
        for u, fname in urls:
            if fname.endswith(".IMG") and not include_img:
                continue
            if fname.endswith(".xml") and not include_xml:
                continue
            if fname.endswith(".png") and not include_png:
                continue
            all_items.append((u, fname))
            
    # Deduplicate by filename
    unique_items = list({fname: (u, fname) for u, fname in all_items}.values())
    print(f"[+] Prepared {len(unique_items)} unique files for download into {dest_dir}.")
    
    if not unique_items:
        return
        
    counts = {"ok": 0, "skipped": 0, "failed": 0}
    t0 = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_file, it, dest_dir): it for it in unique_items}
        for i, future in enumerate(as_completed(futures), 1):
            res = future.result()
            if res == "ok":
                counts["ok"] += 1
            elif res == "skipped":
                counts["skipped"] += 1
            else:
                counts["failed"] += 1
                
            if i % 50 == 0 or i == len(unique_items):
                elapsed = time.time() - t0
                print(f"[{i}/{len(unique_items)}] ok={counts['ok']}, skipped={counts['skipped']}, failed={counts['failed']} ({elapsed:.1f}s)")
                
    total_time = time.time() - t0
    print(f"[+] Finished: {counts['ok']} downloaded, {counts['skipped']} skipped, {counts['failed']} failed in {total_time:.1f}s.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Direct Atlas Downloader via CloudFront")
    parser.add_argument("--site", type=int, default=1, help="Site index (1-8)")
    parser.add_argument("--dest", type=str, default="/tmp/martian_staging", help="Output staging directory")
    parser.add_argument("--product-type", type=str, default="RZS", help="Product type (RZS, RAD)")
    parser.add_argument("--workers", type=int, default=16, help="Download concurrency threads")
    parser.add_argument("--skip-img", action="store_true", help="Skip large .IMG files (download XML & PNG only)")
    args = parser.parse_args()
    
    target_site = SITES[args.site - 1]
    print(f"=== Downloading {target_site['name']} (Sols {target_site['sols']}) ===")
    crawl_and_download(
        date_range_str=target_site["date_range"],
        dest_dir=args.dest,
        product_type=args.product_type,
        max_workers=args.workers,
        include_img=not args.skip_img,
        include_xml=True,
        include_png=True
    )
