#!/usr/bin/env python3
"""Serial 25-unit reconstructed fair-subset CP→CIR batch, including verified completed runs.

The 2 already-completed seed-1 units are verified and reused (no retraining).
Default is a *read-only* plan. --run plus two acknowledgements runs 25 jobs
serially through run_fair_subset_single_seed.py; a rerun verifies/skips completed
units and restarts interrupted stage(s) from epoch 0. NO in-epoch RNG resume.
The batch stops on any failed unit and never marks 25/25 complete prematurely.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
REPRO_ROOT = ROOT / "reproduction"
REPRO_SCRIPTS = REPRO_ROOT / "scripts"
RUNNER = REPRO_SCRIPTS / "run_fair_subset_single_seed.py"
OUT_BASE = REPRO_ROOT / "runs/fair_subset_batch"
STATE_BASE = REPRO_ROOT / "runs/fair_subset_batch_state"
REPORT = STATE_BASE / "fair_subset_batch_status.json"
VARIANTS = ("original", "context", "no_weather", "no_occasion", "no_style")
TASKS = [(variant, seed) for seed in range(1, 6) for variant in VARIANTS]
CP_COLUMNS = ("auc", "fitb_acc")
CIR_COLUMNS = ("recall_at_1", "recall_at_3", "recall_at_5",
               "recall_at_10", "recall_at_30", "recall_at_50")
EXPECTED_QUERIES = 3432


def abort(message):
    raise SystemExit("[BATCH BLOCKED] " + message)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".writing")
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")
    os.replace(temp, path)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def task_path(variant, seed):
    return OUT_BASE / f"{variant}_seed{seed}"


def record_summary(variant, seed, *, verify_files):
    directory = task_path(variant, seed)
    mf = directory / "fair_subset_run_manifest.json"
    if not mf.is_file():
        return {"variant": variant, "seed": seed, "state": "PENDING", "directory": str(directory)}
    manifest = read_json(mf)
    ident = manifest.get("identity") or {}
    if ident.get("variant") != variant or ident.get("seed") != seed:
        abort(f"Wrong variant/seed in existing manifest: {mf}")
    data_sha = ident.get("candidate_id_sha256")
    source_sha = ident.get("source_py_sha256")
    scope_sha = ident.get("cir_3432_scope_report_sha256")
    if not all((data_sha, source_sha, scope_sha)):
        abort(f"Missing source/scope/candidate provenance: {mf}")
    if manifest.get("status") != "passed_candidate_source":
        return {"variant": variant, "seed": seed, "state": "INCOMPLETE",
                "directory": str(directory), "active_stage": manifest.get("active_stage"),
                "candidate_id_sha256": data_sha, "source_py_sha256": source_sha,
                "scope_report_sha256": scope_sha}
    if not all(isinstance(manifest.get(key), dict) for key in ("cp", "cir", "evaluation")):
        abort(f"Passed manifest missing CP/CIR/evaluation: {mf}")
    ev = manifest["evaluation"]
    if ev.get("cir_evaluable") != EXPECTED_QUERIES:
        abort(f"Wrong CIR denominator in passed manifest: {mf}")
    record = {"variant": variant, "seed": seed, "state": "PASSED_CANDIDATE",
              "directory": str(directory), "candidate_id_sha256": data_sha,
              "source_py_sha256": source_sha, "scope_report_sha256": scope_sha,
              "cp_checkpoint_sha256": manifest["cp"]["sha256"],
              "cir_checkpoint_sha256": manifest["cir"]["sha256"],
              "cir_evaluable": ev["cir_evaluable"], "cp": ev["cp"], "cir": ev["cir"]}
    if not verify_files:
        return record

    for name, key in (("cp_best_ckpt.pt", "cp"), ("cir_best_ckpt.pt", "cir")):
        path = directory / name
        if not path.is_file() or sha256(path) != manifest[key]["sha256"]:
            abort(f"Passed checkpoint missing or changed: {path}")
    for name in ("results_cp_fresh_subset.csv", "results_cir_fresh_subset.csv",
                 "detail_cir_fresh_subset.csv"):
        path = directory / name
        if not path.is_file() or sha256(path) != ev.get("sha256", {}).get(name):
            abort(f"Passed evaluation artifact missing or changed: {path}")

    for name, columns, recorded in (
            ("results_cp_fresh_subset.csv", CP_COLUMNS, ev["cp"]),
            ("results_cir_fresh_subset.csv", CIR_COLUMNS, ev["cir"])):
        path = directory / name
        with path.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        if len(rows) != 1 or int(rows[0]["seed"]) != seed or rows[0]["subset_tag"] != "subset":
            abort(f"Metric CSV task/count mismatch: {path}")
        for column in columns:
            if not (0 <= float(rows[0][column]) <= 1) or abs(float(rows[0][column]) - float(recorded[column])) > 1e-12:
                abort(f"Metric CSV differs from recorded result: {path}, {column}")

    detail_path = directory / "detail_cir_fresh_subset.csv"
    pair_ids = set()
    with detail_path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["seed"]) != seed or row["run_tag"] != "none_subset":
                abort(f"Detail row identity mismatch: {detail_path}")
            key = (str(row["set_id"]), str(row["target_item_id"]), str(row["target_item_fg"]))
            if key in pair_ids:
                abort(f"Duplicate CIR evaluation question: {detail_path}, {key}")
            pair_ids.add(key)
    if len(pair_ids) != EXPECTED_QUERIES:
        abort(f"CIR detail has {len(pair_ids)} unique questions, need 3432: {detail_path}")
    canonical = "\n".join("\t".join(v) for v in sorted(pair_ids))
    record["evaluation_query_membership_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    record["evaluation_verified_files"] = True
    return record


def assert_comparable(records):
    complete = [r for r in records if r["state"] == "PASSED_CANDIDATE"]
    if not complete:
        return {"complete": 0, "all_completed_shared_membership": False}
    for key in ("candidate_id_sha256", "source_py_sha256", "scope_report_sha256"):
        uniq = {r[key] if key != "source_py_sha256" else json.dumps(r[key], sort_keys=True)
                for r in complete}
        if len(uniq) != 1:
            abort(f"Completed variant/seed results differ in {key}; cannot compare conditions")
    tested = [r for r in complete if "evaluation_query_membership_sha256" in r]
    if tested and len({r["evaluation_query_membership_sha256"] for r in tested}) != 1:
        abort("3,432 CIR evaluation question membership differs between completed conditions")
    return {"complete": len(complete),
            "verified_complete": len(tested),
            "all_completed_shared_membership": len(tested) == len(complete),
            "evaluation_query_membership_sha256": tested[0]["evaluation_query_membership_sha256"] if tested else None}


def runner_cmd(polyvore, variant, seed, *, execute):
    cmd = [sys.executable, str(RUNNER), "--polyvore-root", str(polyvore),
           "--variant", variant, "--seed", str(seed),
           "--out-dir", str(task_path(variant, seed))]
    if execute:
        cmd.extend(("--run", "--acknowledge-candidate-source"))
    else:
        cmd.append("--dry-run")
    return cmd


def invoke(cmd, logfile, *, stream):
    print("[TASK]", " ".join(cmd), "| LOG", logfile, flush=True)
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with logfile.open("w", encoding="utf-8", buffering=1) as log:
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, bufsize=1)
        try:
            for line in p.stdout:
                log.write(line)
                if stream:
                    print(line, end="", flush=True)
            return p.wait()
        except BaseException:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--polyvore-root", required=True)
    parser.add_argument("--run", action="store_true", help="Launch sequential GPU jobs; default is read-only plan")
    parser.add_argument("--acknowledge-candidate-source", action="store_true")
    parser.add_argument("--acknowledge-long-batch", action="store_true")
    parser.add_argument("--verbose", action="store_true",
                        help="Print full training stdout (always preserved in per-task log)")
    args = parser.parse_args()
    polyvore = Path(args.polyvore_root).expanduser().resolve()
    if not RUNNER.is_file():
        abort(f"Single-task runner missing: {RUNNER}")
    if not (polyvore / "disjoint" / "test.json").is_file():
        abort(f"PO-D data not found: {polyvore}")
    if args.run and not (args.acknowledge_candidate_source and args.acknowledge_long_batch):
        abort("Batch GPU training requires --acknowledge-candidate-source AND --acknowledge-long-batch")
    if args.run:
        # Batch-level lock: no concurrent batch writers/GPUs. Does not prevent a
        # separately launched single-task command; do not start those in parallel.
        STATE_BASE.mkdir(parents=True, exist_ok=True)
        lock = (STATE_BASE / "fair_subset_batch.lock").open("a+")
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            abort("Another subset batch is already running")

    report = {"classification": "reconstructed WOS fair-subset fresh 5-variant x 5-seed CP/CIR batch; NOT historical source exactness",
              "run_mode": "RUN" if args.run else "READ_ONLY_PLAN",
              "started_utc": utc_now(), "updated_utc": utc_now(),
              "total_units": len(TASKS), "per_unit_training": "CP 100 epochs + CIR 100 epochs",
              "completed_units": 0, "status": "PLANNED", "tasks": [],
              "limits": ["Historical fair-subset memberwise ID has not been recovered.",
                         "CP decoder uses the standardized decoder implementation; the historical definition was not preserved.",
                         "Stage-level rerun only; interrupted epochs restart at epoch 0."]}

    def snapshot(status):
        report["tasks"] = [record_summary(v, s, verify_files=args.run) for v, s in TASKS]
        comparability = assert_comparable(report["tasks"])
        report["completed_units"] = comparability["complete"]
        report["comparability"] = comparability
        report["status"] = status
        report["updated_utc"] = utc_now()
        if args.run:
            atomic_write(REPORT, report)
        print(f"[BATCH] {status}; {report['completed_units']}/{len(TASKS)} completed", flush=True)

    snapshot("PREFLIGHT")
    if not args.run:
        print(json.dumps({"mode": "read_only", "complete_from_manifest": report["completed_units"],
                          "remaining": len(TASKS) - report["completed_units"],
                          "candidate_sha256": report["tasks"][0].get("candidate_id_sha256"),
                          "start_command_requires": ["--run", "--acknowledge-candidate-source",
                                                     "--acknowledge-long-batch"],
                          "note": "Existing passed manifests not checkpoint-verified in read-only mode; RUN verifies before skip."},
                         ensure_ascii=False, indent=2))
        return

    for i, (variant, seed) in enumerate(TASKS, start=1):
        report["active_task"] = {"position": i, "variant": variant, "seed": seed}
        atomic_write(REPORT, report)
        logfile = STATE_BASE / "fair_subset_batch_logs" / f"{variant}_seed{seed}.log"
        rc = invoke(runner_cmd(polyvore, variant, seed, execute=True), logfile,
                    stream=args.verbose)
        if rc:
            report["last_failure"] = {"variant": variant, "seed": seed, "returncode": rc,
                                      "log": str(logfile)}
            snapshot("BLOCKED")
            abort(f"{variant}/seed{seed} returned {rc}; inspect {logfile}; "
                  "re-run same batch command to resume from verified stage boundaries")
        report.pop("last_failure", None)
        snapshot("RUNNING")
    report.pop("active_task", None)
    if report["completed_units"] != len(TASKS):
        snapshot("BLOCKED")
        abort("Some units failed completion verification despite zero subprocess exit")
    snapshot("PASSED_CANDIDATE_SOURCE")
    print("[DONE] All 25 reconstructed fair-subset CP/CIR units verified; original historical "
          "subset membership/decoder/source not established. Report:", REPORT, flush=True)


if __name__ == "__main__":
    main()
