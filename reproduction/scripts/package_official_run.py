#!/usr/bin/env python3
"""Package one completed full run as the official result set.

Copies the run's seed-level raw results and summaries into
reproduction/results/raw/<run_id>/ and reproduction/results/summary/<run_id>/
(the layout of the committed reference run), its environment evidence into
reproduction/environment/<run_id>/, and writes:

- results/summary/<run_id>/seed_index.csv: one row per training unit with
  manifest, checkpoint and result hashes, best epochs and runtime;
- results/final_reference_manifest.json: run ID, repository, commit, seeds,
  environment, split checksums, the 70 checkpoint hashes, result locations
  and program hashes;
- results/SHA256SUMS_<run_id>.txt: hashes of every file written above.

Checkpoint hashes are taken from the run's manifests and cross-checked
between manifests; --verify-checkpoints also re-hashes the 70 checkpoint
files. No model is trained or evaluated.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPRO = ROOT / "reproduction"
SEEDS = (1, 2, 3, 4, 5)
MAIN_VARIANTS = ("original", "context")
ABLATION_VARIANTS = ("original", "context", "no_weather", "no_occasion", "no_style")
CONFIG_BUNDLE = ("cp.yaml", "fitb.yaml", "main_hybrid_attention.yaml", "or.yaml")
LOG_TIME = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
SEED_INDEX_FIELDS = (
    "family", "variant", "seed", "status", "git_commit", "training_manifest_sha256",
    "evaluation_manifest_sha256", "documented_config_bundle_sha256",
    "runtime_seconds_from_saved_log_span", "runtime_source_logs", "cp_checkpoint_sha256",
    "cir_checkpoint_sha256", "cp_best_epoch", "cir_best_epoch", "cp_results_sha256",
    "cir_results_sha256", "detail_results_sha256", "candidate_id_sha256",
)


def fail(message: str) -> None:
    raise SystemExit("[PACKAGE BLOCKED] " + message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    if not path.is_file():
        fail(f"missing JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_text(path: Path) -> str:
    if not path.is_file():
        fail(f"missing file: {path}")
    return path.read_text(encoding="utf-8").strip()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


class Packager:
    def __init__(self, run: Path, results_dir: Path, environment_dir: Path) -> None:
        self.run = run
        self.run_id = run.name
        self.results_dir = results_dir
        self.raw = results_dir / "raw" / self.run_id
        self.summary = results_dir / "summary" / self.run_id
        self.environment = environment_dir / self.run_id
        self.written: list[Path] = []

    def copy(self, source: Path, target: Path, required: bool = True) -> bool:
        if not source.is_file():
            if required:
                fail(f"missing run file: {source}")
            return False
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        self.written.append(target)
        return True

    def copy_tree(self, source: Path, target: Path) -> None:
        if not source.is_dir():
            fail(f"missing run directory: {source}")
        for path in sorted(p for p in source.rglob("*") if p.is_file()):
            self.copy(path, target / path.relative_to(source))

    def write(self, target: Path, text: str) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        self.written.append(target)

    # ---- copies -----------------------------------------------------------
    def copy_raw(self) -> None:
        identity = self.raw / "run_identity"
        for name in ("RUN_STATUS.txt", "started_utc.txt", "finished_utc.txt", "git_commit.txt",
                     "git_status_start.txt", "reproduction_summary.json", "reproduction_summary.md"):
            self.copy(self.run / name, identity / name)
        self.copy(self.run / "git_remote.txt", identity / "git_remote.txt", required=False)
        self.copy(self.run / "logs" / "final_terminal_summary.txt",
                  identity / "final_terminal_summary.txt", required=False)
        for variant in MAIN_VARIANTS:
            for seed in SEEDS:
                unit = self.run / "main" / f"{variant}_seed{seed}"
                out = self.raw / "main" / f"{variant}_seed{seed}"
                self.copy(unit / "full_train_manifest.json", out / "full_train_manifest.json")
                for name in ("evaluation_manifest.json", "results_cp.csv", "results_cir.csv"):
                    self.copy(unit / "evaluation" / name, out / name)
        for variant in ABLATION_VARIANTS:
            for seed in SEEDS:
                unit = self.run / "ablation" / "runs" / f"{variant}_seed{seed}"
                out = self.raw / "ablation" / f"{variant}_seed{seed}"
                for name in ("fair_subset_run_manifest.json", "results_cp_fresh_subset.csv",
                             "results_cir_fresh_subset.csv", "detail_cir_fresh_subset.csv"):
                    self.copy(unit / name, out / name)
        fair = self.raw / "fair_subset"
        recon = self.run / "ablation" / "fair_subset_reconstruction"
        self.copy(recon / "fair_subset_reconstruction_manifest.json", fair / "fair_subset_reconstruction_manifest.json")
        self.copy(recon / "SHA256SUMS.txt", fair / "SHA256SUMS.txt")
        self.copy(self.run / "ablation" / "fair_subset_cir_scope_audit.json", fair / "fair_subset_cir_scope_audit.json")
        self.copy(self.run / "ablation" / "fair_subset_or_query_ids.csv", fair / "or_query_ids.csv", required=False)

    def copy_summary(self) -> None:
        main = self.run / "main"
        for name in ("main_reproduction_summary.csv", "main_reproduction_acceptance.json"):
            self.copy(main / "summary" / name, self.summary / "main" / name)
        self.copy_tree(main / "collected", self.summary / "main" / "collected")
        self.copy_tree(self.run / "ablation" / "summary", self.summary / "ablation")
        self.copy_tree(self.run / "statistics", self.summary / "statistics")
        self.copy_tree(self.run / "chapter4", self.summary / "chapter4")
        self.copy_tree(self.run / "secondary" / "results", self.summary / "secondary")
        self.copy_tree(self.run / "environment", self.environment)

    # ---- seed index -------------------------------------------------------
    def runtime(self, first_log: Path, last_log: Path) -> int:
        def stamp(path: Path, last: bool) -> datetime:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            for line in (reversed(lines) if last else lines):
                match = LOG_TIME.match(line)
                if match:
                    return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
            fail(f"no timestamp in log: {path}")
        return int((stamp(last_log, True) - stamp(first_log, False)).total_seconds())

    def config_bundle_sha(self, commit: str) -> str:
        configs = REPRO / "configs"
        changed = subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", commit, "--",
                                  *[str(configs / n) for n in CONFIG_BUNDLE]])
        if changed.returncode != 0:
            fail(f"reproduction/configs differ from the run commit {commit}")
        # SHA-256 of the SHA256SUMS-style lines of the four configs, sorted by name.
        lines = "".join(f"{sha256(configs / n)}  {n}\n" for n in CONFIG_BUNDLE)
        return hashlib.sha256(lines.encode("utf-8")).hexdigest()

    def seed_index(self, commit: str, verify_checkpoints: bool) -> list[dict]:
        bundle = self.config_bundle_sha(commit)
        rows = []
        for variant in MAIN_VARIANTS:
            for seed in SEEDS:
                unit = self.run / "main" / f"{variant}_seed{seed}"
                train, evaluation = unit / "full_train_manifest.json", unit / "evaluation" / "evaluation_manifest.json"
                tm, em = read_json(train), read_json(evaluation)
                for key in ("cp_checkpoint_sha256", "cir_checkpoint_sha256"):
                    if tm[key] != em[key]:
                        fail(f"{unit}: {key} differs between training and evaluation manifests")
                for name, key in (("results_cp.csv", "results_cp_sha256"), ("results_cir.csv", "results_cir_sha256")):
                    if sha256(unit / "evaluation" / name) != em[key]:
                        fail(f"{unit}: {name} no longer matches its evaluation manifest")
                if verify_checkpoints:
                    for name, key in (("cp_best_ckpt.pt", "cp_checkpoint_sha256"), ("cir_best_ckpt.pt", "cir_checkpoint_sha256")):
                        if sha256(unit / name) != tm[key]:
                            fail(f"{unit}: {name} does not match its manifest")
                rows.append({
                    "family": "main", "variant": variant, "seed": seed, "status": em["status"],
                    "git_commit": commit, "training_manifest_sha256": sha256(train),
                    "evaluation_manifest_sha256": sha256(evaluation),
                    "documented_config_bundle_sha256": bundle,
                    "runtime_seconds_from_saved_log_span": self.runtime(unit / "cp_full_train.log", unit / "cir_full_train.log"),
                    "runtime_source_logs": "cp_full_train.log;cir_full_train.log",
                    "cp_checkpoint_sha256": tm["cp_checkpoint_sha256"], "cir_checkpoint_sha256": tm["cir_checkpoint_sha256"],
                    "cp_best_epoch": tm["cp_best_epoch"], "cir_best_epoch": tm["cir_best_epoch"],
                    "cp_results_sha256": em["results_cp_sha256"], "cir_results_sha256": em["results_cir_sha256"],
                    "detail_results_sha256": "", "candidate_id_sha256": "",
                })
        for variant in ABLATION_VARIANTS:
            for seed in SEEDS:
                unit = self.run / "ablation" / "runs" / f"{variant}_seed{seed}"
                manifest = unit / "fair_subset_run_manifest.json"
                fm = read_json(manifest)
                if verify_checkpoints:
                    for name, stage in (("cp_best_ckpt.pt", "cp"), ("cir_best_ckpt.pt", "cir")):
                        if sha256(unit / name) != fm[stage]["sha256"]:
                            fail(f"{unit}: {name} does not match its manifest")
                rows.append({
                    "family": "candidate_ablation", "variant": variant, "seed": seed, "status": fm["status"],
                    "git_commit": commit, "training_manifest_sha256": sha256(manifest),
                    "evaluation_manifest_sha256": "", "documented_config_bundle_sha256": bundle,
                    "runtime_seconds_from_saved_log_span": self.runtime(unit / "cp_train.log", unit / "cir_train.log"),
                    "runtime_source_logs": "cp_train.log;cir_train.log",
                    "cp_checkpoint_sha256": fm["cp"]["sha256"], "cir_checkpoint_sha256": fm["cir"]["sha256"],
                    "cp_best_epoch": fm["cp"]["best_epoch"], "cir_best_epoch": fm["cir"]["best_epoch"],
                    "cp_results_sha256": sha256(unit / "results_cp_fresh_subset.csv"),
                    "cir_results_sha256": sha256(unit / "results_cir_fresh_subset.csv"),
                    "detail_results_sha256": sha256(unit / "detail_cir_fresh_subset.csv"),
                    "candidate_id_sha256": fm["identity"]["candidate_id_sha256"],
                })
        if len(rows) != 35 or len({r["cp_checkpoint_sha256"] for r in rows} | {r["cir_checkpoint_sha256"] for r in rows}) != 70:
            fail("expected 35 units with 70 distinct checkpoint hashes")
        return rows

    # ---- final manifest ---------------------------------------------------
    def final_manifest(self, commit: str, rows: list[dict], verified: bool) -> dict:
        env = read_json(self.run / "environment" / "environment_validation.json")
        runtime = env["torch_runtime"]
        cudnn = int(runtime["cudnn_version"])
        started, finished = read_text(self.run / "started_utc.txt"), read_text(self.run / "finished_utc.txt")
        seconds = int((datetime.fromisoformat(finished.replace("Z", "+00:00"))
                       - datetime.fromisoformat(started.replace("Z", "+00:00"))).total_seconds())
        status_lines = read_text(self.run / "git_status_start.txt").splitlines()
        branch = status_lines[0].removeprefix("## ").split("...")[0] if status_lines else "not_available"
        remote = self.run / "git_remote.txt"
        scope = read_json(self.run / "ablation" / "fair_subset_cir_scope_audit.json")
        evidence = read_json(self.run / "statistics" / "statistical_evidence.json")
        fair_summary = read_json(self.run / "ablation" / "summary" / "fresh_fair_subset_5seed_summary.json")

        def sums(path: Path) -> dict:
            return {name: digest for digest, name in
                    (line.split("  ", 1) for line in read_text(path).splitlines() if line)}

        def summary_path(run_relative: str) -> str:
            mapped = run_relative.replace("ablation/summary/", "ablation/", 1)
            return rel(self.summary / mapped)

        return {
            "official_run_id": self.run_id,
            "classification": "official 2026 clean-room run; formal numeric source of the TORS revision",
            "repository": {
                "url": read_text(remote) if remote.is_file() else "not_available",
                "branch": branch, "commit": commit, "git_status_start": status_lines,
            },
            "execution": {
                "run_status": read_text(self.run / "RUN_STATUS.txt"), "started_utc": started,
                "finished_utc": finished, "duration_seconds": seconds,
                "duration_hms": f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}",
            },
            "training_units": {"main": 10, "fair_subset_ablation": 25, "total": len(rows)},
            "seeds": list(SEEDS),
            "environment": {
                "python": env["python"].split()[0], "platform": env.get("platform"),
                "torch_cuda": runtime.get("torch_cuda_version"),
                "cudnn": f"{cudnn // 10000}.{cudnn % 10000 // 100}.{cudnn % 100}",
                "gpu": sorted({d["name"] for d in runtime.get("devices", [])}),
                "packages": {name: info.get("installed_version") for name, info in env.get("packages", {}).items()},
            },
            "splits": {
                "main_manifests_sha256": sums(self.run / "frozen_splits" / "SHA256SUMS.txt"),
                "fair_subset_sha256": sums(self.run / "ablation" / "fair_subset_reconstruction" / "SHA256SUMS.txt"),
                "or_query_membership_sha256": scope.get("evaluable_query_membership_sha256",
                                                        fair_summary.get("evaluation_query_membership_sha256")),
                "or_evaluable_queries": scope["observed_preserved_evaluator_semantics"]["evaluable_questions"],
            },
            "checkpoints": {
                "hash_source": "re-hashed checkpoint files" if verified else "recorded in the run's manifests",
                "units": [{k: r[k] for k in ("family", "variant", "seed", "cp_checkpoint_sha256",
                                             "cir_checkpoint_sha256", "cp_best_epoch", "cir_best_epoch")}
                          for r in rows],
            },
            "results": {
                "raw": rel(self.raw), "summary": rel(self.summary),
                "seed_index": rel(self.summary / "seed_index.csv"),
                "environment": rel(self.environment),
                "hash_manifest": rel(self.results_dir / f"SHA256SUMS_{self.run_id}.txt"),
                "thesis_tables": {table: summary_path(path)
                                  for table, path in evidence.get("thesis_tables", {}).items()},
            },
            "programs": {
                "git_commit": commit,
                "statistics_script_sha256": evidence.get("analysis_script_sha256"),
                "config_bundle_sha256": rows[0]["documented_config_bundle_sha256"],
            },
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--results-dir", type=Path, default=REPRO / "results")
    parser.add_argument("--environment-dir", type=Path, default=REPRO / "environment")
    parser.add_argument("--verify-checkpoints", action="store_true",
                        help="Also re-hash the 70 checkpoint files (reads about 1.9 GB).")
    args = parser.parse_args()

    run = args.run_root.expanduser().resolve()
    if read_text(run / "RUN_STATUS.txt") != "PASSED":
        fail(f"run is not PASSED: {run}")
    packager = Packager(run, args.results_dir.expanduser().resolve(), args.environment_dir.expanduser().resolve())
    for target in (packager.raw, packager.summary, packager.environment):
        if target.exists():
            fail(f"refusing to overwrite existing output: {target}")
    commit = read_text(run / "git_commit.txt")

    rows = packager.seed_index(commit, args.verify_checkpoints)
    packager.copy_raw()
    packager.copy_summary()
    seed_index = packager.summary / "seed_index.csv"
    seed_index.parent.mkdir(parents=True, exist_ok=True)
    with seed_index.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SEED_INDEX_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    packager.written.append(seed_index)
    manifest = packager.final_manifest(commit, rows, args.verify_checkpoints)
    packager.write(packager.results_dir / "final_reference_manifest.json",
                   json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    sums = packager.results_dir / f"SHA256SUMS_{packager.run_id}.txt"
    lines = sorted(f"{sha256(p)}  {p.resolve().relative_to(packager.results_dir)}"
                   for p in packager.written if p.resolve().is_relative_to(packager.results_dir))
    sums.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[PACKAGED] {packager.run_id}: {len(rows)} units, {len(packager.written)} files")
    print(f"[PACKAGED] final manifest: {packager.results_dir / 'final_reference_manifest.json'}")
    print(f"[PACKAGED] hash manifest: {sums} ({len(lines)} files)")


if __name__ == "__main__":
    main()
