import json
import time
import urllib.request

SEARCH_API = "https://pds-imaging.jpl.nasa.gov/api/search/atlas/_search"

SITES = [
    {
        "idx": 1,
        "id": "site1",
        "name": "Landing Site Bedrock (Octavia E. Butler Landing)",
        "sols": "15-30",
        "date_range": "[2021-03-06 TO 2021-03-22]",
        "geology": "Crater floor basaltic bedrock, flat pavement, landing thruster blast zone",
        "lighting_value": "Baseline calibration sequences, diverse solar elevations from high noon to late afternoon"
    },
    {
        "idx": 2,
        "id": "site2",
        "name": "South Seitah Dunes & Megaripples",
        "sols": "195-225",
        "date_range": "[2021-09-07 TO 2021-10-09]",
        "geology": "Aeolian ripple fields, basaltic sand dunes, layered outcrops (Bastide & Brac)",
        "lighting_value": "Extreme shadow dynamics across sharp dune ridges and micro-relief under grazing sun"
    },
    {
        "idx": 3,
        "id": "site3",
        "name": "Delta Front Scarp & Enchanted Lake",
        "sols": "420-445",
        "date_range": "[2022-04-26 TO 2022-05-23]",
        "geology": "Massive delta front cliffs, steep scarps, layered mudstone and boulder conglomerate",
        "lighting_value": "Vertical cliff face shadow movement, deep self-shadowing, multi-scale terrain occlusion"
    },
    {
        "idx": 4,
        "id": "site4",
        "name": "Skinner Ridge (Delta Top)",
        "sols": "455-475",
        "date_range": "[2022-06-01 TO 2022-06-22]",
        "geology": "Delta top sedimentary layered rocks, fine mudstones, polygonal fracture patterns",
        "lighting_value": "Stationary multi-sol core sampling stop, pristine multi-illumination angle phase coverage"
    },
    {
        "idx": 5,
        "id": "site5",
        "name": "Tenby Mudstones (Amalik)",
        "sols": "520-545",
        "date_range": "[2022-08-07 TO 2022-09-02]",
        "geology": "Fine-grained deltaic mudstones, river deposits, sample cache depot preparation",
        "lighting_value": "Dense diurnal solar phase cycle across stationary rover orientations"
    },
    {
        "idx": 6,
        "id": "site6",
        "name": "Margin Carbonate Unit (Jurabi Point)",
        "sols": "955-985",
        "date_range": "[2023-10-28 TO 2023-11-28]",
        "geology": "Stromatolite-like carbonate formations, silica-rich rocks along the inner crater margin",
        "lighting_value": "Highly reflective light-toned carbonate minerals with unique specular and scattering properties"
    },
    {
        "idx": 7,
        "id": "site7",
        "name": "Bright Angel Channel & Neretva Vallis",
        "sols": "1165-1190",
        "date_range": "[2024-05-30 TO 2024-06-26]",
        "geology": "Ancient river paleochannel, water-transported rounded boulders, bright jagged outcrop blocks",
        "lighting_value": "Complex multi-albedo scene with high contrast between bright bedrock blocks and dark channel sands"
    },
    {
        "idx": 8,
        "id": "site8",
        "name": "Crater Rim Lookout (Aurora Peak)",
        "sols": "1830-1860",
        "date_range": "[2026-04-14 TO 2026-05-16]",
        "geology": "High-altitude Jezero crater rim viewpoint, panoramic lookouts over the entire crater basin",
        "lighting_value": "Long-range atmospheric haze, distant crater rim horizon, sunset/sunrise solar phase variations"
    }
]

def query_stats(date_range_str, product_type="RZS"):
    qs = (
        f"_exists_:gather.uri AND "
        f"(gather.common.spacecraft:perseverance AND "
        f"gather.common.instrument:(MCZ_LEFT OR MCZ_RIGHT) AND "
        f"gather.common.kind:regular AND "
        f"gather.pds_archive.bundle_id:mars2020_mastcamz_ops_calibrated AND "
        f"gather.common.product_type:{product_type} AND "
        f"gather.time.start_time:{date_range_str})"
    )
    
    payload = {
        "query": {"query_string": {"query": qs}},
        "from": 0,
        "size": 10,
        "_source": ["archive.size", "gather.pds_archive.related"]
    }
    req = urllib.request.Request(
        SEARCH_API,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        
    total_hits = data.get("hits", {}).get("total", {}).get("value", 0)
    
    # Calculate sample file sizes to get accurate average
    sample_sizes = []
    for hit in data.get("hits", {}).get("hits", []):
        src = hit.get("_source", {})
        arch_size = src.get("archive", {}).get("size")
        rel = src.get("gather", {}).get("pds_archive", {}).get("related", {})
        src_size = rel.get("src", {}).get("size") or arch_size or 5500000
        lbl_size = rel.get("label", {}).get("size") or 88000
        brw_size = rel.get("browse", {}).get("size") or 2500000
        sample_sizes.append(src_size + lbl_size + brw_size)
        
    avg_size_bytes = sum(sample_sizes) / len(sample_sizes) if sample_sizes else (8.5 * 1024 * 1024)
    return total_hits, avg_size_bytes

results = []
print("Querying all 8 sites from NASA PDS Atlas API...")
for s in SITES:
    total, avg_bytes = query_stats(s["date_range"], "RZS")
    total_files = total * 3  # (.IMG, .xml, .png)
    total_gb = (total * avg_bytes) / (1024.0 ** 3)
    results.append({
        "idx": s["idx"],
        "name": s["name"],
        "sols": s["sols"],
        "date_range": s["date_range"],
        "frames": total,
        "total_files": total_files,
        "total_gb": round(total_gb, 2),
        "geology": s["geology"],
        "lighting_value": s["lighting_value"]
    })
    print(f"Site {s['idx']}: {s['name']} (Sols {s['sols']}) -> {total} frames ({total_files} files) ~ {total_gb:.2f} GB")
    time.sleep(0.2)

with open("c:/MachineLearning/MartianImages/site_stats.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved stats to site_stats.json")
