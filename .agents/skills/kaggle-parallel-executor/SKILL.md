---
name: kaggle-parallel-executor
description: >-
  Multi-account headless Kaggle GPU/CPU execution runbook. Orchestrates parallel kernel dispatching,
  private dataset authentication mounting, live status tracking, and Hugging Face artifact
  synchronization without interrupting the primary interactive SSH session.
---

# Kaggle Multi-Account Parallel Executor Skill

This skill teaches the agent how to leverage multiple auxiliary Kaggle accounts (`gourav`, `gourav_ju`, `johan`, `nexus`) to dispatch and execute parallel GPU/CPU jobs headlessly via the Kaggle API.

The primary interactive notebook session connected via Cloudflare SSH remains completely uninterrupted for exploratory analysis and debugging, while heavy preprocessing, embedding generation, or model training jobs run asynchronously in parallel across cloud workers.

--------------------------------------------------------------------------------

## 1. Core Architecture & Workflow

```
[Local / Antigravity Orchestrator]
       │
       ├──> Ensure private <account>/hf-auth dataset exists on Kaggle
       ├──> Build staging directory with main.py & kernel-metadata.json
       ├──> Push headless kernel via kaggle CLI (with KAGGLE_CONFIG_DIR set)
       │
       ▼
[Kaggle Cloud Worker (GPU / CPU)]
       │
       ├──> Read HF token from mounted /kaggle/input/hf-auth/hf_token.txt
       ├──> Pull dependencies/inputs from gouravroy2110/exemplum/<project_folder>/...
       ├──> Execute compute workload (PyTorch, CatBoost, XGBoost, etc.)
       └──> Upload outputs directly to gouravroy2110/exemplum/<project_folder>/...
       │
       ▼
[Local / Antigravity Orchestrator]
       │
       ├──> Poll kernel status (KernelWorkerStatus.RUNNING -> COMPLETE)
       └──> Retrieve completed artifacts from Hugging Face
```

--------------------------------------------------------------------------------

## 2. Credential Management & Multi-Account Layout

Account credentials reside in `C:\Users\goura\.kaggle\<account>\`:
- `gourav` -> `gouravroy2006` (`kaggle.json`)
- `gourav_ju` -> `thenamelessmonster` (`kaggle.json`)
- `johan` -> `johanliebert04041975` (`kaggle.json`)
- `nexus` -> `nexus` (`access_token`)

To switch between accounts when invoking the Kaggle CLI or Python API, set:
```python
env = os.environ.copy()
env["KAGGLE_CONFIG_DIR"] = r"C:\Users\goura\.kaggle\<account>"
```

--------------------------------------------------------------------------------

## 3. Zero-Leak Credential Mounting Pattern

**Critical Rule:** Never hardcode credentials, inject token strings into script code, or print tokens in output logs.

### Private Auth Dataset Mount:
1. Kaggle script kernels only package the single declared `code_file`. Local files in the push directory are NOT placed in `/kaggle/working`.
2. To provide the Hugging Face token safely, each account maintains a strictly private dataset (`<username>/hf-auth`) containing `hf_token.txt`.
3. In `kernel-metadata.json`, include:
   ```json
   "dataset_sources": ["<username>/hf-auth"]
   ```
4. In the worker script, read the token from the standard mount point:
   ```python
   token_path = "/kaggle/input/hf-auth/hf_token.txt"
   if os.path.exists(token_path):
       with open(token_path, "r", encoding="utf-8") as f:
           hf_token = f.read().strip()
   ```

--------------------------------------------------------------------------------

## 4. Kernel Staging & Metadata Schema

Each dispatched job requires a staging directory containing:
- `main.py` — The standalone worker script.
- `kernel-metadata.json` — Kernel configuration:

```json
{
  "id": "<username>/<job-slug>",
  "title": "<Job Title>",
  "code_file": "main.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": "true",
  "enable_gpu": "true",
  "enable_tpu": "false",
  "enable_internet": "true",
  "dataset_sources": ["<username>/hf-auth"],
  "competition_sources": [],
  "kernel_sources": [],
  "model_sources": []
}
```

*Note:* `"enable_internet": "true"` is mandatory so the runner can reach Hugging Face.

--------------------------------------------------------------------------------

## 5. Bidirectional Hugging Face Artifact Synchronization

All permanent inputs and outputs are scoped under the project directory prefix (`<project_folder>/`, e.g., `AmazonMLChallenge/`):

### A. Worker Startup (Pulling Inputs / Weights):
```python
from huggingface_hub import hf_hub_download

input_file = hf_hub_download(
    repo_id="gouravroy2110/exemplum",
    filename="<project_folder>/artifacts/input_data.parquet",
    repo_type="dataset",
    token=hf_token
)
```

### B. Worker Completion (Pushing Output Artifacts):
```python
from huggingface_hub import HfApi

api = HfApi(token=hf_token)
api.upload_file(
    path_or_fileobj="output_model.json",
    path_in_repo="<project_folder>/models/output_model.json",
    repo_id="gouravroy2110/exemplum",
    repo_type="dataset",
    commit_message=f"Persist model from headless runner {job_slug}"
)
```

--------------------------------------------------------------------------------

## 6. Tactical Insights & Best Practices

1. **Kaggle CLI Script Isolation:**
   When `kernel_type: "script"` is used, Kaggle CLI discards other files in the staging folder. Only the file declared in `code_file` is uploaded to `/kaggle/src/script.py`. Auxiliary dependencies must be attached as datasets or installed at runtime.
2. **Dataset Default Privacy:**
   `kaggle datasets create` defaults to private (`is_private: True`). Do not supply `-u` or `--public` for credential or sensitive datasets.
3. **Avoid Shell Escaping Issues in PowerShell:**
   Do not run complex multi-line inline Python code strings in PowerShell (`$env:` and quotes frequently break). Instead, run dedicated `.py` dispatcher scripts or use Python `subprocess.run(..., env=env)`.
4. **Unbuffered Python Logging:**
   Always run long-running monitoring scripts with unbuffered output (`python -u` or `sys.stdout.reconfigure(line_buffering=True)`) so status checks stream in real time.
5. **Reusable Dispatch Tool:**
   Use the bundled runner script at `scripts/runner.py`:
   ```powershell
   python .agents/skills/kaggle-parallel-executor/scripts/runner.py list-accounts
   python .agents/skills/kaggle-parallel-executor/scripts/runner.py dispatch --account gourav --slug sample-job --title "Sample Job" --script my_worker.py
   ```
