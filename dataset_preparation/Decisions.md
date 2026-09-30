# Dataset Decisions: Product Types, Staging Architecture & Site Priorities

## 1. Product Type Decision

**Selected Default:** `RZS` (Synthesized RGB Color). May incorporate `RAD` (Linear Radiance) in subsequent photometric passes if required.

### Crisp Definitions:
* **RZS (RGB Color Synthesized):** Calibrated natural human-vision color representation, balanced to standard display RGB for instant visualization, inspection, and generative Martian World Modeling.
* **RAD (Calibrated Radiance):** Sensor-calibrated linear photon flux ($W / m^2 / sr / nm$) with electronic noise and flat-fields removed, strictly preserving physical shadows and surface reflectance for photometric 4DGS neural rendering.

---

## 2. Ingestion Architecture: Two-Stage CloudFront Pipeline

### The Problem:
* Directly querying `pds-imaging.jpl.nasa.gov` or downloading from JPL on Kaggle triggers an **AWS WAF 403 Forbidden** because JPL blocks datacenter IPs.
* Manually clicking, paginating, and exporting `.sh` wget scripts through the Atlas web interface is slow and prone to truncated downloads.

### The Solution (Two-Stage Pipeline):
1. **Stage 1 (Local Metadata Query):**
   * The local workstation queries NASA's Atlas Elasticsearch REST API (`https://pds-imaging.jpl.nasa.gov/api/search/atlas/_search`) using exact Lucene JSON queries.
   * Extracts every matching record and its complete PDS4 file triplets:
     1. `.IMG`: Calibrated full-resolution science image product.
     2. `.xml`: Complete PDS4 label containing pointing geometry, solar azimuth, solar elevation, rover orientation quaternion, local coordinates, and Local True Solar Time (LTST).
     3. `.png`: Full-resolution browse preview for the web studio.
   * Maps each PDS4 URI to NASA's direct CloudFront CDN (`https://d1ejlg980osaur.cloudfront.net/m20/`):
     * If `release_id_num == 0` (mission baseline / earlier Sols): maps to `/m20/cumulative/`
     * If `release_id_num > 0` (recent releases): maps to `/m20/r{release_id_num}/`
   * Generates a single clean manifest (`siteX_manifest.txt`).

2. **Stage 2 (Kaggle CloudFront Ingestion):**
   * The manifest is transferred to Kaggle via SSH/SCP.
   * `fast_pds_downloader.py` uses a 16-thread `ThreadPoolExecutor` to pull files directly from CloudFront into `/tmp/martian_staging`.
   * **Performance:** Downloads at **136+ MB/s (~45 files/sec)** with zero 403 errors, automatic retries, and skip-if-exists resume support.
   * Leverages Kaggle's **1.1 TB scratch disk** on `/tmp/` without exhausting the 20 GB `/kaggle/working` limit.
   * Symlinked to `/kaggle/working/martian_staging` for direct shell navigation.

---

## 3. Site Inventory & Sizing Analysis

Every frame is downloaded as a complete triplet: **`[.IMG + .xml + .png]`**. No metadata is omitted.

| Rank | Site | Sol Range | UTC Date Range | Total Frames | Total Files (Triplet) | Approx Size | Primary Geologic & Lighting Value |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **P1** | **Site 1: Landing Bedrock** *(Octavia E. Butler)* | 15–30 | 2021-03-06 to 2021-03-22 | 624 | 1,872 | **3.09 GB** | Flat basaltic floor, thruster blast zone. Baseline calibration & clean stationary sun sweeps. *(Already staged!)* |
| **P2** | **Site 3: Delta Front Scarp** *(Enchanted Lake)* | 420–445 | 2022-04-26 to 2022-05-23 | 2,212 | 6,636 | **29.25 GB** | Massive delta cliffs, multi-scale sedimentary layering. Extreme cliff face self-shadowing & morning-to-evening occlusion. |
| **P3** | **Site 5: Tenby Mudstones** *(Amalik)* | 520–545 | 2022-08-07 to 2022-09-02 | 2,140 | 6,420 | **16.18 GB** | Fine-grained mudstones, sample depot prep. High-density diurnal time sequences from stationary rover poses. |
| **P4** | **Site 8: Crater Rim Lookout** *(Aurora Peak)* | 1830–1860 | 2026-04-14 to 2026-05-16 | 2,473 | 7,419 | **22.25 GB** | Elevated overlook of entire Jezero crater basin. Long-range atmospheric haze and horizon solar phase scattering. |
| **P5** | **Site 6: Margin Carbonates** *(Jurabi Point)* | 955–985 | 2023-10-28 to 2023-11-28 | 2,948 | 8,844 | **39.08 GB** | Light-toned carbonate outcrops. High-albedo mineral reflections, unique specular highlights and scattering properties. |
| **P6** | **Site 7: Bright Angel Channel** *(Neretva Vallis)* | 1165–1190 | 2024-05-30 to 2024-06-26 | 3,478 | 10,434 | **39.30 GB** | Ancient river channel deposit, jagged light-toned blocks and gravel bars. High-contrast multi-albedo scene. |
| **P7** | **Site 4: Skinner Ridge** *(Delta Top)* | 455–475 | 2022-06-01 to 2022-06-22 | 3,727 | 11,181 | **46.86 GB** | Delta top sedimentary outcrops, polygonal fractures. Stationary core-sampling sequences across multiple sols. |
| **P8** | **Site 2: South Séítah Dunes** | 195–225 | 2021-09-07 to 2021-10-09 | 5,969 | 17,907 | **85.81 GB** | Basaltic sand dunes & megaripples. Incredible ripple shadows under grazing sun, but largest dataset size. |

**Total Across All 8 Sites:** **23,571 frames** (70,713 files) $\approx$ **281.8 GB**.

---

## 4. Priority Ranking Rationale

### Tier A: Immediate Core Foundations (Sites 1, 3, 5) — $\approx 48.5\text{ GB}$
* **Site 1 (3.1 GB - Staged):** Clean bedrock benchmark to calibrate photometric NeRF/4DGS and ensure the pipeline correctly parses sun angles and coordinate frames.
* **Site 3 (29.3 GB):** Highest priority 3D visual feature on Mars. Vertical cliff geometry produces the most dramatic shadow sweeps and multi-illumination test cases for 4DGS.
* **Site 5 (16.2 GB):** Compact footprint with rich stationary rover time series at different solar elevations.

### Tier B: Macro Horizon & Mineral Diversity (Sites 8, 6) — $\approx 61.3\text{ GB}$
* **Site 8 (22.3 GB):** High-elevation panoramas critical for learning long-range atmospheric scattering, sky domes, and global Jezero crater context for the Martian World Model.
* **Site 6 (39.1 GB):** Unlocks light-toned carbonate minerals with drastically different bidirectional reflectance distribution functions (BRDF).

### Tier C: High-Density Staging Batches (Sites 7, 4, 2) — $\approx 172.0\text{ GB}$
* **Site 7 (39.3 GB) & Site 4 (46.9 GB):** Extensive rock fields and delta top core sites.
* **Site 2 (85.8 GB):** The single largest site corpus (85+ GB). Can be selectively sampled or staged after Tier A is verified in the web studio.

---

## 5. Exact Atlas Lucene Query Template

```text
_exists_:gather.uri AND (
  gather.common.spacecraft:perseverance AND 
  gather.common.instrument:(MCZ_LEFT OR MCZ_RIGHT) AND 
  gather.common.kind:regular AND 
  gather.pds_archive.bundle_id:mars2020_mastcamz_ops_calibrated AND 
  gather.common.product_type:RZS AND 
  gather.time.start_time:[<START_DATE> TO <END_DATE>]
)
```
