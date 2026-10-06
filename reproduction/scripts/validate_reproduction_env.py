#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import importlib.metadata as md
import json
from pathlib import Path
import platform
import sys

EXPECTED = {
    "numpy": "2.0.2",
    "torch": "2.9.1",
    "torchvision": "0.24.1",
    "transformers": "4.57.1",
    "scikit-learn": "1.6.1",
    "einops": "0.8.1",
    "Pillow": "11.3.0",
    "tqdm": "4.67.1",
    "ipdb": "0.13.13",
}

IMPORT_NAME = {
    "numpy": "numpy",
    "torch": "torch",
    "torchvision": "torchvision",
    "transformers": "transformers",
    "scikit-learn": "sklearn",
    "einops": "einops",
    "Pillow": "PIL",
    "tqdm": "tqdm",
    "ipdb": "ipdb",
}

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=None)
args = ap.parse_args()

report = {
    "python": sys.version,
    "platform": platform.platform(),
    "packages": {},
    "torch_runtime": {},
}

hard_fail = False
for dist, expected in EXPECTED.items():
    item = {"expected_archived_version": expected}
    try:
        installed = md.version(dist)
        item["installed_version"] = installed
        item["exact_version_match"] = installed == expected
    except Exception as e:
        item["installed_version"] = None
        item["exact_version_match"] = False
        item["metadata_error"] = repr(e)
        hard_fail = True
    try:
        importlib.import_module(IMPORT_NAME[dist])
        item["import_ok"] = True
    except Exception as e:
        item["import_ok"] = False
        item["import_error"] = repr(e)
        hard_fail = True
    report["packages"][dist] = item

try:
    import torch
    report["torch_runtime"] = {
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "devices": [
            {
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "capability": list(torch.cuda.get_device_capability(i)),
            }
            for i in range(torch.cuda.device_count())
        ] if torch.cuda.is_available() else [],
        "float16_autocast_available": bool(torch.cuda.is_available()),
    }
except Exception as e:
    report["torch_runtime_error"] = repr(e)
    hard_fail = True

exact = all(x.get("exact_version_match") for x in report["packages"].values())
imports = all(x.get("import_ok") for x in report["packages"].values())
cuda_ok = bool(report.get("torch_runtime", {}).get("cuda_available"))
torch_cuda_build = report.get("torch_runtime", {}).get("torch_cuda_version")
report["summary"] = {
    "all_required_imports_ok": imports,
    "all_versions_exactly_match_archived_freeze": exact,
    "checkpoint_eval_environment_status": (
        "exact_archived_package_versions_with_cuda_runtime" if exact and imports and cuda_ok
        else "exact_archived_package_versions_but_no_cuda_runtime" if exact and imports
        else "imports_ok_but_version_drift" if imports
        else "missing_or_broken_dependencies"
    ),
    "cuda_runtime_available": cuda_ok,
    "torch_is_cuda_build": torch_cuda_build is not None,
    "note": (
        "The archived requirements.txt is a later/full workspace freeze and is not yet proven "
        "to be the exact 2025 training environment. Version mismatch is therefore recorded, "
        "not silently corrected."
    ),
}

payload = json.dumps(report, ensure_ascii=False, indent=2)
print(payload)
if args.out:
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(payload + "\n", encoding="utf-8")
sys.exit(2 if hard_fail else 0)
