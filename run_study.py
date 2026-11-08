"""Configure and launch one fresh F1 study, with a read-only preflight by default."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import stat
import subprocess
import sys
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent
SOURCE_DIRS = ("authority", "fuse_addon", "fuse_ane", "fuse_credit",
               "fuse_ig", "fuse_paired", "fuse_patch", "vendor")
FIELDS = {"run_id", "task", "group", "search_seed", "cpu", "runtime",
          "image", "credential_file", "api_base_url"}


def load_config(path, root=ROOT):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict) or set(config) != FIELDS:
        raise ValueError("CONFIG_FIELDS_MUST_MATCH_EXAMPLE")
    for key in ("run_id", "group"):
        if not isinstance(config[key], str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", config[key]):
            raise ValueError("INVALID_" + key.upper())
    if config["task"] not in ("tsp", "cvrp"):
        raise ValueError("TASK_MUST_BE_TSP_OR_CVRP")
    if type(config["search_seed"]) is not int or not 0 <= config["search_seed"] < 2**32:
        raise ValueError("SEARCH_SEED_MUST_BE_UINT32")
    if type(config["cpu"]) is not int or config["cpu"] < 0:
        raise ValueError("CPU_MUST_BE_NONNEGATIVE_INTEGER")
    for key in ("runtime", "image", "credential_file", "api_base_url"):
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError("INVALID_" + key.upper())
    for key in ("runtime", "credential_file"):
        value = Path(config[key]).expanduser()
        # Keep symlinks visible to credential validation.
        config[key] = str((root / value).absolute() if not value.is_absolute() else value)
    url = urlsplit(config["api_base_url"])
    if (url.scheme != "https" or not url.hostname or url.username or url.password
            or url.query or url.fragment or config["api_base_url"].endswith("/")):
        raise ValueError("API_BASE_URL_MUST_BE_HTTPS_WITHOUT_TRAILING_SLASH")
    return config


def preflight(config, root=ROOT):
    """Inspect prerequisites; never send a model request or execute a candidate."""
    errors, identity = [], {}
    if platform.system() != "Linux":
        errors.append("Full experiments require Linux; offline examples support macOS.")
    if hasattr(os, "sched_getaffinity") and config["cpu"] not in os.sched_getaffinity(0):
        errors.append("The configured CPU is outside this process's allowed CPU set.")
    if not (Path(config["runtime"]) / "bin/python").is_file():
        errors.append("Configure a container-compatible Python environment in runtime.")
    if not (root / "fuse_patch/rpc_worker.py").is_file():
        errors.append("The candidate RPC worker is missing.")

    panel_dir = root / "panels" / config["task"] / config["group"]
    panels = sorted(panel_dir.glob("feedback-*.json"))
    if not panels:
        errors.append("Dataset panels: coming soon. Supply panels/<task>/<group>/feedback-*.json before a run.")
    else:
        ids = []
        try:
            from fuse_patch import rpc_worker
            for path in panels:
                rows = json.loads(path.read_text(encoding="utf-8"))["instances"]
                for row in rows:
                    if not isinstance(row["instance_id"], str) or not row["instance_id"]:
                        raise ValueError("INVALID_INSTANCE_ID")
                    command = {"task": "TSP-Construct" if config["task"] == "tsp" else "CVRP-Construct",
                               "execution_seed": row["execution_seed"], "distance_matrix": row["distance_matrix"]}
                    if len(command["distance_matrix"]) != (50 if config["task"] == "tsp" else 51):
                        raise ValueError("UNEXPECTED_TASK_SIZE")
                    if config["task"] == "cvrp":
                        command["demands"] = row["demands"]
                        if (type(row["capacity"]) is not int or row["capacity"] != 40 or row["depot"] != 0
                                or row["demands"][0] != 0 or any(not 1 <= d <= 9 for d in row["demands"][1:])):
                            raise ValueError("INVALID_CVRP_CAPACITY_OR_DEPOT")
                    rpc_worker.validate_init(command)
                    ids.append(row["instance_id"])
            if len(ids) != 64 or len(set(ids)) != 64:
                raise ValueError("FEEDBACK_REQUIRES_64_UNIQUE_INSTANCES")
            identity["panels"] = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in panels}
        except (OSError, KeyError, IndexError, TypeError, ValueError):
            errors.append("Feedback panels must contain 64 valid, uniquely identified task instances.")

    if urlsplit(config["api_base_url"]).hostname.endswith(".invalid"):
        errors.append("Replace api_base_url with the configured model service endpoint.")
    credential = Path(config["credential_file"])
    if not credential.is_file() or credential.is_symlink():
        errors.append("Provide a regular, non-symlink credential file outside the source allowlist.")
    elif stat.S_IMODE(credential.stat().st_mode) != 0o600:
        errors.append("Set the credential file's permissions to 0600.")
    else:
        # Bind parsing locally without retaining or printing the API key.
        from fuse_ig import recovery_credentials
        previous = recovery_credentials.BASE_URL
        try:
            recovery_credentials.BASE_URL = config["api_base_url"]
            recovery_credentials.parse(credential.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            errors.append("Credential fields, endpoint, or model do not match the configured protocol.")
        finally:
            recovery_credentials.BASE_URL = previous

    if shutil.which("docker") is None:
        errors.append("Docker is required for candidate isolation.")
    else:
        try:
            result = subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", config["image"]],
                                    capture_output=True, text=True, timeout=15)
            image_id = result.stdout.strip()
            if result.returncode or not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
                errors.append("The configured Docker image is unavailable; prepare a compatible local image.")
            else:
                identity["image"] = image_id
        except (OSError, subprocess.TimeoutExpired):
            errors.append("Docker image inspection failed or timed out.")
    for name in ("engineering", "runs", "STOP_DISPATCH.json", "SEALED.json", "SEARCH_LOCK.json", "SELECTION_LOCK.json"):
        if (root / name).exists():
            errors.append("Existing study state detected; use a fresh checkout for a new study.")
            break
    return errors, identity


def source_manifest(root=ROOT):
    paths = list(root.glob("*.py"))
    for name in SOURCE_DIRS:
        paths.extend(p for p in (root / name).rglob("*") if p.is_file() and p.suffix in (".py", ".txt"))
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def launch(config, identity, root=ROOT):
    """Create fresh local records, then call the existing F1 search implementation."""
    engineering = root / "engineering"
    engineering.mkdir()  # Exclusive study setup; no silent resume or overwrite.
    runtime = {"image": identity["image"], "runtime": config["runtime"], "cpus": [config["cpu"]]}
    records = {
        "RUNTIME.json": runtime,
        "SOURCE_FREEZE.json": {"files": source_manifest(root), "panels": identity["panels"]},
        "AUTHORITY.json": {"run_id": config["run_id"], "method": "F1", "explicit_launch": True,
                           "evaluation_budget": 200},
        "RUN_CONFIG.json": config,
    }
    for name, value in records.items():
        with (engineering / name).open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
    env = os.environ.copy()
    env.update(DECADE_EXECUTION_ROOT=str(root), DECADE_CREDENTIAL_FILE=config["credential_file"],
               DECADE_API_BASE_URL=config["api_base_url"], PYTHONDONTWRITEBYTECODE="1")
    spec = {"id": config["run_id"], "task": config["task"], "group": config["group"],
            "method": "F1", "search_seed": config["search_seed"]}
    code = ("import json, sys; from main_runtime import search; "
            "result = search(json.loads(sys.argv[1]), int(sys.argv[2])); "
            "print(json.dumps(result, indent=2)); "
            "sys.exit(0 if result['normal_completion'] else 1)")
    return subprocess.run([sys.executable, "-c", code, json.dumps(spec), str(config["cpu"])],
                          cwd=root, env=env).returncode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/f1.example.json")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Check prerequisites without starting a study (default).")
    mode.add_argument("--run", action="store_true", help="Start a fresh study; this sends model requests and evaluates candidates.")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        errors, identity = preflight(config)
    except (OSError, TypeError, ValueError) as exc:
        print("Configuration check failed:", type(exc).__name__, file=sys.stderr)
        return 2
    if errors:
        print("Prerequisites to complete:")
        for error in errors:
            print("-", error)
        return 1
    print("Prerequisite checks passed; the container runtime and model service have not been exercised.")
    return launch(config, identity) if args.run else 0


if __name__ == "__main__":
    raise SystemExit(main())
