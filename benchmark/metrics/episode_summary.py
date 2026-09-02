#!/usr/bin/env python3
"""
episode_summary.py — aggregate multi-run eval results for a single episode

Reads the raw eval records (run_XX/metrics/skill_YY_ep00.json, produced by
benchmark.skill_eval) and aggregates across runs into episode_summary.json.

Directory layout (outer level = run index; each skill runs once per run):
  {log_dir}/run_01/metrics/skill_00_ep00.json
  {log_dir}/run_02/metrics/skill_00_ep00.json  ...

Metrics:
  per_run[i]  : single run → TSCR / cTSCR / FFI / full_success / stsr_by_{field}
  aggregate   : across runs → mean/std TSCR, cTSCR, full_SR, per_skill_SR, stsr_by_{field}

Usage (from the repo root):
  python -m benchmark.metrics.episode_summary \
      --log_dir output/eval/task_a/episode_111 \
      --episode xxxxxxxx \
      [--num_runs N]   # default 0 = auto-scan the number of run_XX dirs
      [--eval_config data/skill_eval_configs/task_a/episode_111.yaml]
      [--group_by type,skill_description]
"""

import argparse
import glob
import json
import os
import statistics as st
from collections import Counter, defaultdict


# ── Utilities ──────────────────────────────────────────────────────────────────

def load_skill_meta(eval_config: str) -> dict:
    """Load skill_id → skill fields (type, skill_description, ...) from the eval_config YAML."""
    if not eval_config or not os.path.exists(eval_config):
        return {}
    try:
        import yaml
        with open(eval_config, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return {
            s["id"]: {k: v for k, v in s.items() if k != "bddl_goal"}
            for s in cfg.get("skills", [])
        }
    except Exception as e:
        print(f"[WARN] load_skill_meta: {e}")
        return {}


def parse_run(run_dir: str) -> list:
    """Parse metrics/skill_XX_ep00.json under a single run dir, returning the per-skill results."""
    results = []
    for mf in sorted(glob.glob(os.path.join(run_dir, "metrics", "skill_*_ep00.json"))):
        with open(mf, encoding="utf-8") as f:
            data = json.load(f)
        results.append({
            "skill_id": data["skill_id"],
            "success":  bool(data["success"]),
            "n_steps":  data.get("n_steps"),
        })
    return sorted(results, key=lambda x: x["skill_id"])


def count_runs(log_dir: str) -> int:
    """Count the run_XX dirs under log_dir (used as the default num_runs)."""
    return len(glob.glob(os.path.join(log_dir, "run_[0-9][0-9]")))


def run_metrics(skill_results: list):
    """Compute TSCR, cTSCR and the first failed skill_id for a single run."""
    M = len(skill_results)
    if M == 0:
        return 0.0, 0.0, None
    n_succ = sum(1 for r in skill_results if r["success"])
    k_star = 0
    for r in skill_results:
        if r["success"]:
            k_star += 1
        else:
            break
    ffi = next((r["skill_id"] for r in skill_results if not r["success"]), None)
    return n_succ / M, k_star / M, ffi


def compute_stsr(skill_results: list, skill_meta: dict, field: str = "type") -> dict:
    """Group skills by an arbitrary YAML field and compute per-group success rates.

    field: any field name from the skills config, e.g. 'type', 'skill_description'.
    Returns {field_value: {n, n_success, SR}}.
    """
    count = defaultdict(int)
    success = defaultdict(int)
    for r in skill_results:
        val = skill_meta.get(r["skill_id"], {}).get(field, f"<unknown_{field}>")
        count[val] += 1
        if r["success"]:
            success[val] += 1
    return {
        val: {
            "n":         count[val],
            "n_success": success[val],
            "SR":        round(success[val] / count[val], 4) if count[val] else 0,
        }
        for val in sorted(count)
    }


# ── Main aggregation ──────────────────────────────────────────────────────────

def aggregate_episode(log_dir: str, episode: str = "", num_runs: int = 0,
                      eval_config: str = "", group_fields: list = None) -> dict:
    """Aggregate the multi-run results of a single episode, write episode_summary.json
    and return the summary dict.

    When num_runs <= 0, the number of run_XX dirs is auto-scanned.
    When the eval_config YAML is unavailable, stsr_by_{field} is skipped
    (STSR omitted, no error).
    Returns None when there is no valid data.
    """
    group_fields = group_fields or ["type"]
    skill_meta = load_skill_meta(eval_config)
    have_meta = bool(skill_meta)

    if num_runs <= 0:
        num_runs = count_runs(log_dir)
        if num_runs == 0:
            print(f"[WARN] no run_XX dirs found in {log_dir}")
            return None
        print(f"[INFO] auto-detected num_runs={num_runs} in {log_dir}")

    # ── Parse each run ──────────────────────────────────────────────────────────
    all_runs = []
    for ri in range(1, num_runs + 1):
        run_dir = os.path.join(log_dir, f"run_{ri:02d}")
        if not os.path.isdir(run_dir):
            print(f"[WARN] run dir not found, skipping: {run_dir}")
            continue
        skill_results = parse_run(run_dir)
        if not skill_results:
            print(f"[WARN] no skill results in {run_dir}, skipping")
            continue

        tcr, ctcr, ffi = run_metrics(skill_results)

        run_entry = {
            "run":           ri,
            "skill_results": skill_results,
            "TSCR":           round(tcr,  4),
            "cTSCR":          round(ctcr, 4),
            "FFI":           ffi,
            "full_success":  all(r["success"] for r in skill_results),
        }
        if have_meta:
            for gf in group_fields:
                run_entry[f"stsr_by_{gf}"] = compute_stsr(skill_results, skill_meta, gf)
        all_runs.append(run_entry)

    if not all_runs:
        print(f"[WARN] No run data found in {log_dir}")
        return None

    M     = len(all_runs[0]["skill_results"])
    n_run = len(all_runs)

    # ── Cross-run aggregation: TSCR / cTSCR ──────────────────────────────────────────────
    tcrs  = [r["TSCR"]  for r in all_runs]
    ctcrs = [r["cTSCR"] for r in all_runs]
    mean_tcr  = sum(tcrs)  / n_run
    mean_ctcr = sum(ctcrs) / n_run
    std_tcr   = st.pstdev(tcrs)  if n_run > 1 else 0.0
    std_ctcr  = st.pstdev(ctcrs) if n_run > 1 else 0.0

    # full_SR: fraction of runs where all M skills succeed (the strictest task success metric)
    full_sr  = sum(1 for r in all_runs if r["full_success"]) / n_run
    ffi_mode = Counter(r["FFI"] for r in all_runs).most_common(1)[0][0]

    # ── Cross-run aggregation: per-skill SR ─────────────────────────────────────────────
    skill_ids = sorted({r["skill_id"] for run in all_runs for r in run["skill_results"]})
    per_skill_sr = []
    for sid in skill_ids:
        succs = [r["success"] for run in all_runs for r in run["skill_results"]
                 if r["skill_id"] == sid]
        per_skill_sr.append({
            "skill_id":  sid,
            "n_runs":    len(succs),
            "n_success": sum(succs),
            "SR":        round(sum(succs) / len(succs), 4) if succs else 0,
        })

    # ── Cross-run aggregation: stsr_by_{field} ──────────────────────────────────────────
    agg_stsr_map = {}
    if have_meta:
        all_skill_results = [r for run in all_runs for r in run["skill_results"]]
        for gf in group_fields:
            agg_stsr_map[gf] = compute_stsr(all_skill_results, skill_meta, gf)
    elif group_fields:
        print(f"[WARN] eval_config unavailable for episode {episode!r}, "
              f"skipping STSR (group_by: {','.join(group_fields)})")

    # ── Write out ──────────────────────────────────────────────────────────────
    summary = {
        "episode":       episode,
        "num_runs":      n_run,
        "M":             M,
        "TSCR":           {"mean": round(mean_tcr,  4), "std": round(std_tcr,  4)},
        "cTSCR":          {"mean": round(mean_ctcr, 4), "std": round(std_ctcr, 4)},
        "full_SR":       round(full_sr, 4),
        "FFI_mode":      ffi_mode,
        "per_skill_SR":  per_skill_sr,
        **{f"stsr_by_{gf}": stsr for gf, stsr in agg_stsr_map.items()},
        "per_run":       all_runs,
    }

    out_path = os.path.join(log_dir, "episode_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"  Episode {episode} ({n_run} runs / {M} skills):")
    print(f"    TSCR      : {mean_tcr:.4f} ± {std_tcr:.4f}")
    print(f"    cTSCR     : {mean_ctcr:.4f} ± {std_ctcr:.4f}")
    print(f"    full_SR  : {full_sr:.4f}  "
          f"({sum(1 for r in all_runs if r['full_success'])}/{n_run} fully succeeded)")
    print(f"    FFI_mode : {ffi_mode}")
    for gf, stsr in agg_stsr_map.items():
        print(f"    STSR by {gf}:")
        for val, v in stsr.items():
            print(f"      {val:<30} {v['n_success']:>2}/{v['n']:<2}  {v['SR']*100:>5.1f}%")
    print(f"  → {out_path}")
    return summary


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate per-run eval results into episode_summary.json"
    )
    parser.add_argument("--log_dir", required=True,
                        help="Episode log dir (contains run_01/, run_02/, ...)")
    parser.add_argument("--num_runs", type=int, default=0,
                        help="number of runs to aggregate (default 0 = auto-scan the number of run_XX dirs)")
    parser.add_argument("--episode", default="")
    parser.add_argument("--eval_config", default="",
                        help="skill_eval_configs/.../episode_XX.yaml, source of skill metadata")
    parser.add_argument("--group_by", default="type",
                        help="YAML fields to group STSR by, comma-separated; "
                             "e.g. type, skill_description (default: type)")
    args = parser.parse_args()

    group_fields = [f.strip() for f in args.group_by.split(",") if f.strip()]
    aggregate_episode(
        log_dir=args.log_dir,
        episode=args.episode,
        num_runs=args.num_runs,
        eval_config=args.eval_config,
        group_fields=group_fields,
    )


if __name__ == "__main__":
    main()
