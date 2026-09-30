# AGENT.md — Remote Kaggle GPU Bridge
**Local SSH Target:** `kaggle` (configured in `~/.ssh/config`)

---

## 1. Remote Kaggle GPU Tunneling (Cloudflare + SSH Key Auth)

Kaggle notebooks provide 2x NVIDIA Tesla T4 (15GB each) or P100 GPUs, but are behind a strict firewall. We bridge local Antigravity execution to Kaggle via a secure, passwordless **Cloudflare Quick Tunnel (`cloudflared`)** authenticated with an **Ed25519 SSH key**.

### Session Startup Protocol:
1. **User Starts Session in Kaggle Notebook:**
   Runs the notebook cell that installs OpenSSH, writes the public key to `/root/.ssh/authorized_keys`, and launches `cloudflared tunnel --url tcp://localhost:22`.
   The cell outputs:
   ```text
   SUCCESS! Tunnel Hostname: <random-subdomain>.trycloudflare.com
   ```
2. **Updating the Hostname in `~/.ssh/config`:**
   - **Agent Automation:** The user can paste the hostname into chat (`Here is the tunnel: <hostname>`). The agent reads `C:\Users\goura\.ssh\config` and updates the `HostName` line under `Host kaggle`.
   - **Manual Alternative:** The user can edit `~/.ssh/config` directly.
   - **Local SSH Config Structure (`C:\Users\goura\.ssh\config`):**
     ```ssh-config
     Host kaggle
         HostName <active-subdomain>.trycloudflare.com
         User root
         IdentityFile ~/.ssh/id_kaggle
         ProxyCommand "C:/Program Files (x86)/cloudflared/cloudflared.exe" access tcp --hostname %h
         StrictHostKeyChecking no
         UserKnownHostsFile /dev/null
         RemotePlatform linux
     ```

---

## 2. Remote Command Execution Rules for Agents

When issuing commands to Kaggle over SSH from Windows PowerShell:

1. **Always Use the `-n` Flag:**
   Windows OpenSSH will hang on non-interactive commands waiting for stdin EOF unless detached:
   ```powershell
   ssh -n kaggle "<remote_command>"
   ```
2. **Kaggle Python Runtime Location:**
   The default Kaggle Python binary with CUDA & PyTorch installed is at:
   ```bash
   /usr/local/bin/python
   ```
3. **GPU / CUDA PATH Initialization & `nvidia-smi` Fix:**
   Non-login SSH sessions and clean root shells on Kaggle have a minimal default `PATH` (`/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin`). On Kaggle, `nvidia-smi` is located in `/opt/bin`, and CUDA binaries are in `/usr/local/cuda/bin` and `/usr/local/nvidia/bin`. Without these in `PATH`, running `nvidia-smi` or CUDA tools directly in SSH returns `command not found`.
   
   To persist this for all interactive and non-interactive SSH sessions, add them to `/root/.bashrc` and `/etc/profile.d/kaggle_gpu.sh`:
   ```bash
   echo "export PATH=/opt/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:\$PATH" | cat - /root/.bashrc > /tmp/.bashrc && mv /tmp/.bashrc /root/.bashrc
   echo "export PATH=/opt/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:\$PATH" > /etc/profile.d/kaggle_gpu.sh
   chmod +x /etc/profile.d/kaggle_gpu.sh
   ```
   *Verification command:*
   ```powershell
   ssh -n kaggle "nvidia-smi; /usr/local/bin/python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count())'"
   ```

---

## 3. Fast File Sync & Incremental Code Edits (via `scp`)

Never re-upload heavy datasets or full repositories for small code tweaks. Use fast `scp` transfers over the Cloudflare tunnel:

### A. Initializing or Transferring a Test Folder:

A sample code is given below:

```powershell
scp -r "c:\MachineLearning\MartianImages\Dynamic_4DGS_Solar_Lighting\remote_test" kaggle:/kaggle/working/
```

### B. Syncing Small Edits (1-2 seconds):
When modifying an individual file locally (e.g., `scene/gaussian_model.py`):

A sample program is given below:

```powershell
scp "c:\MachineLearning\MartianImages\Dynamic_4DGS_Solar_Lighting\fork_Martial_World_Model_longfeiLi_gitRepo\scene\gaussian_model.py" kaggle:/kaggle/working/fork_Martial_World_Model_longfeiLi_gitRepo/scene/
```

### C. Directory Syncing Rules:
- **Never sync** `.git`, `__pycache__`, raw datasets, or `.venv` over SSH.
- Keep large Martian image datasets attached via `/kaggle/input/` (read-only) and point training scripts to output to `/kaggle/working/`.

---

## 4. Persistent Storage & Hugging Face Dataset Backup

Kaggle environments handle storage differently depending on execution mode:
- **Interactive Main Session (`/kaggle/working`):** In the primary interactive notebook session connected via Cloudflare SSH, session persistence can be enabled for `/kaggle/working` (20 GB). This interactive notebook remains dedicated to rapid exploratory analysis, data inspection, and script prototyping without disruption.
- **Temporary Ephemeral Buffer (`/tmp`):** `/tmp` provides a large temporary overlay buffer (8.0 TB) for intermediate matrix computations, temporary tokenization buffers, and raw extraction, but is always wiped when a session terminates.
- **Cross-Session & Headless Job Hub (Hugging Face):** To dispatch heavy or long-running ready jobs in parallel across auxiliary Kaggle accounts without halting or interfering with the main interactive notebook, all inputs, models, embeddings, and generated artifacts are synchronized via Hugging Face.

### Remote Hugging Face Repository & Project Scoping:
* **Target Dataset:** `https://huggingface.co/datasets/gouravroy2110/exemplum`
* **Local / Kaggle HF User:** `gouravroy2110`
* **Authentication:** Authenticated via `~/.cache/huggingface/token` (never printed, logged, or hardcoded).
* **Project Folder Scoping Pattern:**
  To support reusing this configuration across multiple projects without cross-contamination, all artifacts, models, logs, and outputs MUST be scoped under a top-level directory named after the active workspace root folder (`<project_folder>/`).
  
  *Example (for this project, `<project_folder>` is `AmazonMLChallenge`):*
  - `<project_folder>/artifacts/` — intermediate tables, candidate pairs, tokenizers.
  - `<project_folder>/models/` — trained weights, configuration files, checkpoints.
  - `<project_folder>/submissions/` — competition submission files.
  - `<project_folder>/logs/` — training metrics, verification reports, evaluation sheets.

### Bidirectional Persistence Patterns:
1. **Pulling Dependencies / Pretrained Assets at Worker Startup:**
   ```python
   from huggingface_hub import hf_hub_download

   # Example: pulling from AmazonMLChallenge/artifacts/
   local_path = hf_hub_download(
       repo_id="gouravroy2110/exemplum",
       filename="<project_folder>/artifacts/candidate_pairs.parquet",  # e.g., AmazonMLChallenge/artifacts/...
       repo_type="dataset",
   )
   ```

2. **Uploading Artifacts / Models:**
   ```python
   from huggingface_hub import HfApi

   api = HfApi()
   # Example: uploading to AmazonMLChallenge/models/
   api.upload_file(
       path_or_fileobj="models/phase2_xgboost.json",
       path_in_repo="<project_folder>/models/phase2_xgboost.json",  # e.g., AmazonMLChallenge/models/...
       repo_id="gouravroy2110/exemplum",
       repo_type="dataset",
       commit_message="Persist Phase 2 XGBoost model",
   )
   ```

3. **Uploading Entire Output Directories:**
   ```python
   api.upload_folder(
       folder_path="output/artifacts",
       path_in_repo="<project_folder>/artifacts",  # e.g., AmazonMLChallenge/artifacts
       repo_id="gouravroy2110/exemplum",
       repo_type="dataset",
   )
   ```

### Parallel Remote Headless Execution (Multi-Account Scaling):
For dispatching non-interactive GPU/CPU jobs across auxiliary Kaggle accounts (`gourav`, `gourav_ju`, `johan`, `nexus`) without touching the primary SSH notebook, refer to the project skill:
**[`kaggle-parallel-executor`](file:///.agents/skills/kaggle-parallel-executor/SKILL.md)**.
- Authentication tokens must be mounted read-only via private Kaggle datasets (`<username>/hf-auth`).
- Headless scripts must pull inputs from and push outputs directly to `gouravroy2110/exemplum/<project_folder>/`.

---

## 5. Teardown & Session Cleanup Protocol

When the training or debugging session concludes:
1. **Push Needed Artifacts to Hugging Face:** Ensure any critical models, embeddings, or candidate sets are uploaded to `gouravroy2110/exemplum`.
2. **Kill Background Processes on Kaggle:**
   ```powershell
   ssh -n kaggle "pkill -f cloudflared; service ssh stop; rm -f /root/.ssh/authorized_keys"
   ```
3. **Clear Outputs in Kaggle UI:** Clear the output of the cell that printed the `trycloudflare.com` URL so the ephemeral domain is not stored in the notebook draft.
4. **Stop Session:** Stop the Kaggle notebook session to release GPU quota.

---

## 6. Gradio Interactive Demo Port-Forwarding (Port 8080)

```powershell
ssh -N -L 8080:localhost:8080 kaggle
```
*(Or directly using the Cloudflare domain: `ssh -N -L 8080:localhost:8080 <active-subdomain>.trycloudflare.com`)*

---

## 7. Strict Coding Agent Guidelines & Guardrails

To prevent erroneous results, guarantee scientific rigor, and ensure close monitoring, the following rules MUST be strictly adhered to by any coding agent:

1. **No Unauthorized Hardcoding:** The coding agent is STRICTLY PROHIBITED from using any hardcoded data, magic numbers, or arbitrary threshold values based solely on its own reasoning. If the agent believes hardcoding is absolutely necessary, it MUST present the reasoning and values to the user and await explicit user validation before writing the code.
2. **Mandatory Planning & Approval Phase:** Before writing, modifying, or executing any code for a new task, the agent MUST first prepare a clear, step-by-step plan. Execution or coding can only begin AFTER the user explicitly approves the plan.
3. **Mandatory Small-Scale Testing:** All new pipeline stages (blocking, matching, feature engineering) MUST be initially tested on a small, representative slice of the data (e.g., `rows=5000`). The agent must verify execution success, memory usage, and basic metric sanity before proposing to run on the full dataset.
4. **Reproducibility First:** The agent must explicitly set random seeds (`np.random.seed(42)`, `torch.manual_seed(42)`, `random_seed=42`) for all randomized operations to ensure that all experiments are fully deterministic and reproducible.
5. **Metric Logging:** All scripts must log clear validation metrics (Recall, Candidate Size, Precision, F_0.5) to standard output so the user can continuously monitor the performance trajectory.
6. **Strict Prohibition on Unscientific Hyperbole & Definitive Terminology:** The agent is STRICTLY PROHIBITED from using any hyperbolic, marketing, or definitive words—including but not limited to "god", "final", "superior", "master plan", "champion", "winning", "perfect", or "ultimate"—when describing models, strategies, pipelines, or outputs. All technical discussions, reports, and logs must remain strictly empirical, objective, and measured.
7. **Modular, Disjoint Script Architecture (Single Responsibility Principle):** The agent MUST write clean, disjoint scripts where each script serves a single, specific function in the pipeline.
   - **Shared Libraries & Modules:** All reusable logic (e.g. text normalization, string similarity features, address/number parsing, metric calculation, I/O streaming) MUST reside in modular utility files (e.g. `utils.py`, `features.py`, `data_loader.py`).
   - **Disjoint Executables:** Operational scripts (e.g. `preprocess.py`, `train_xgboost.py`, `train_catboost.py`, `evaluate.py`, `inference.py`) MUST simply import and call the necessary functions from the shared modules.
   - **No Monolithic Kitchen-Sink Scripts:** Never combine data extraction, feature engineering, multiple model training routines, and final submission generation inside one giant script. Maintain strict separation of concerns across the codebase.
