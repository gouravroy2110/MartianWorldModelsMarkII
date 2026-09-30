"""
Kaggle Parallel Executor Utility.
Handles multi-account headless GPU/CPU script dispatching, status polling, and Hugging Face artifact integration.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from typing import Dict, List, Optional
from huggingface_hub import HfApi, hf_hub_download

# Force unbuffered output for live monitoring
sys.stdout.reconfigure(line_buffering=True)

KAGGLE_KEYS_BASE = os.path.expanduser(r"~\.kaggle")
HF_TOKEN_PATH = os.path.expanduser(r"~\.cache\huggingface\token")
HF_REPO_ID = "gouravroy2110/exemplum"


def get_available_accounts() -> Dict[str, Dict[str, str]]:
    """Scan ~/.kaggle/ subdirectories for account credentials."""
    accounts = {}
    if not os.path.exists(KAGGLE_KEYS_BASE):
        return accounts

    for folder in os.listdir(KAGGLE_KEYS_BASE):
        folder_path = os.path.join(KAGGLE_KEYS_BASE, folder)
        if not os.path.isdir(folder_path):
            continue
        json_file = os.path.join(folder_path, "kaggle.json")
        if os.path.exists(json_file):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                accounts[folder] = {
                    "username": data.get("username", folder),
                    "config_dir": folder_path,
                    "type": "kaggle.json"
                }
            except Exception as e:
                print(f"Warning: Failed reading {json_file}: {e}", file=sys.stderr)
        elif os.path.exists(os.path.join(folder_path, "access_token")):
            accounts[folder] = {
                "username": folder,
                "config_dir": folder_path,
                "type": "access_token"
            }
    return accounts


def ensure_hf_auth_dataset(account_name: str, config_dir: str, username: str) -> str:
    """Ensure a private hf-auth dataset exists on the specified Kaggle account."""
    if not os.path.exists(HF_TOKEN_PATH):
        raise FileNotFoundError(f"Hugging Face token not found at {HF_TOKEN_PATH}")

    with open(HF_TOKEN_PATH, "r", encoding="utf-8") as f:
        token = f.read().strip()

    ds_ref = f"{username}/hf-auth"
    env = os.environ.copy()
    env["KAGGLE_CONFIG_DIR"] = config_dir

    # Check if dataset already exists
    res = subprocess.run(
        ["python", "-m", "kaggle", "datasets", "status", ds_ref],
        env=env,
        capture_output=True,
        text=True
    )
    if res.returncode == 0 and "ready" in res.stdout:
        return ds_ref

    # Create dataset if missing
    staging_dir = os.path.join(os.path.dirname(__file__), f"_tmp_ds_{account_name}")
    os.makedirs(staging_dir, exist_ok=True)
    try:
        with open(os.path.join(staging_dir, "hf_token.txt"), "w", encoding="utf-8") as f:
            f.write(token)
        meta = {
            "title": f"hf-auth-{username}",
            "id": ds_ref,
            "licenses": [{"name": "CC0-1.0"}]
        }
        with open(os.path.join(staging_dir, "dataset-metadata.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        create_res = subprocess.run(
            ["python", "-m", "kaggle", "datasets", "create", "-p", staging_dir],
            env=env,
            capture_output=True,
            text=True
        )
        print(f"Created private auth dataset {ds_ref}: {create_res.stdout.strip()}")
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)

    return ds_ref


def dispatch_kernel(
    account_key: str,
    slug: str,
    title: str,
    script_path: str,
    gpu: bool = True,
    additional_datasets: Optional[List[str]] = None,
    stage_dir: Optional[str] = None
) -> str:
    """Stage and push a headless script kernel to Kaggle."""
    accounts = get_available_accounts()
    if account_key not in accounts:
        raise ValueError(f"Unknown account '{account_key}'. Available: {list(accounts.keys())}")

    acc = accounts[account_key]
    username = acc["username"]
    config_dir = acc["config_dir"]

    # Ensure auth dataset is available
    auth_ds = ensure_hf_auth_dataset(account_key, config_dir, username)

    # Prepare staging directory
    if stage_dir is None:
        stage_dir = os.path.abspath(f"./_stage_{account_key}_{slug}")
    if os.path.exists(stage_dir):
        shutil.rmtree(stage_dir)
    os.makedirs(stage_dir, exist_ok=True)

    # Copy worker script to main.py
    shutil.copy(script_path, os.path.join(stage_dir, "main.py"))

    dataset_sources = [auth_ds]
    if additional_datasets:
        dataset_sources.extend(additional_datasets)

    meta = {
        "id": f"{username}/{slug}",
        "title": title,
        "code_file": "main.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true" if gpu else "false",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": dataset_sources,
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": []
    }

    with open(os.path.join(stage_dir, "kernel-metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    env = os.environ.copy()
    env["KAGGLE_CONFIG_DIR"] = config_dir

    print(f"Pushing {meta['id']} from account [{account_key}]...")
    res = subprocess.run(
        ["python", "-m", "kaggle", "kernels", "push", "-p", stage_dir],
        env=env,
        capture_output=True,
        text=True
    )
    print("Output:", res.stdout.strip())
    if res.stderr:
        print("Stderr:", res.stderr.strip())

    return meta["id"]


def poll_kernels(kernel_account_map: Dict[str, str], poll_interval: int = 15):
    """Monitor kernel statuses until all reach terminal status."""
    accounts = get_available_accounts()
    statuses = {kid: "queued" for kid in kernel_account_map}
    start_time = time.time()

    print("=" * 60)
    print(f"Monitoring {len(kernel_account_map)} Kaggle Kernels...")
    print("=" * 60)

    while True:
        all_done = True
        for kid, acc_key in kernel_account_map.items():
            curr = statuses[kid]
            if any(term in curr.upper() for term in ["COMPLETE", "ERROR", "CANCEL"]):
                continue

            cfg_dir = accounts[acc_key]["config_dir"]
            env = os.environ.copy()
            env["KAGGLE_CONFIG_DIR"] = cfg_dir

            res = subprocess.run(
                ["python", "-m", "kaggle", "kernels", "status", kid],
                env=env,
                capture_output=True,
                text=True
            )
            out = res.stdout.strip()
            statuses[kid] = out
            elapsed = round(time.time() - start_time)
            print(f"[{elapsed}s] {kid} -> {out}")

            if not any(term in out.upper() for term in ["COMPLETE", "ERROR", "CANCEL"]):
                all_done = False

        if all_done:
            print("All monitored kernels reached terminal status.")
            break
        time.sleep(poll_interval)


def main():
    parser = argparse.ArgumentParser(description="Kaggle Parallel Headless Execution Helper")
    subparsers = parser.add_subparsers(dest="command")

    # List accounts
    subparsers.add_parser("list-accounts", help="List configured Kaggle accounts")

    # Dispatch
    p_disp = subparsers.add_parser("dispatch", help="Dispatch a headless script kernel")
    p_disp.add_argument("--account", required=True, help="Account key name (e.g. gourav, gourav_ju, johan)")
    p_disp.add_argument("--slug", required=True, help="Kernel slug")
    p_disp.add_argument("--title", required=True, help="Kernel title")
    p_disp.add_argument("--script", required=True, help="Path to Python script to execute")
    p_disp.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU")
    p_disp.add_argument("--dataset", action="append", help="Additional dataset slug to mount")

    args = parser.parse_args()

    if args.command == "list-accounts":
        accs = get_available_accounts()
        print(f"Found {len(accs)} Kaggle accounts:")
        for k, v in accs.items():
            print(f"  - {k}: username='{v['username']}' (dir: {v['config_dir']})")
    elif args.command == "dispatch":
        kid = dispatch_kernel(
            account_key=args.account,
            slug=args.slug,
            title=args.title,
            script_path=args.script,
            gpu=not args.cpu,
            additional_datasets=args.dataset
        )
        print(f"Dispatched kernel: {kid}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
