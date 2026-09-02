#!/usr/bin/env python3
"""
task_summary.py — cross-episode aggregation for a single task → task_summary.json

Reads {log_root}/episode_*/episode_summary.json (produced by episode_summary.py)
and aggregates across episodes into task_summary.json.

Three levels of metrics:
  per_run[i]   : run i aggregated over all episodes → mean_TSCR / full_SR / stsr_by_type
  per_episode  : per-trajectory aggregation across runs (from episode_summary.json)
  aggregate    : across all runs and episodes → Mean/Pooled-TSCR, full_SR, STSR

Usage (from the repo root):
  python -m benchmark.metrics.task_summary \
      --log_root output/eval/task_a \
      --task task_a \
      [--configs_root data/skill_eval_configs/task_a]
"""

import argparse
import glob
import json
import os
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

try:
    from .episode_summary import load_skill_meta
except ImportError:  # direct script invocation (python benchmark/metrics/task_summary.py)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from episode_summary import load_skill_meta

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ── Utilities ──────────────────────────────────────────────────────────────────

def merge_stsr(stsr_list: list) -> dict:
    """Merge multiple stsr dicts, accumulating n / n_success and recomputing SR."""
    merged = defaultdict(lambda: {"n": 0, "n_success": 0})
    for stsr in stsr_list:
        for t, v in stsr.items():
            merged[t]["n"]         += v["n"]
            merged[t]["n_success"] += v["n_success"]
    return {
        t: {
            "n":         merged[t]["n"],
            "n_success": merged[t]["n_success"],
            "SR":        round(merged[t]["n_success"] / merged[t]["n"], 4)
                         if merged[t]["n"] else 0,
        }
        for t in sorted(merged)
    }


def _episode_yaml(configs_root: str, task: str, ep_id: str) -> str:
    """Episode YAML path; falls back to the task-level single YAML ({configs_root}/../{task}.yaml) when missing."""
    yaml_path = os.path.join(configs_root, f"episode_{ep_id}.yaml")
    if not os.path.exists(yaml_path):
        fallback = os.path.join(os.path.dirname(configs_root), f"{task}.yaml")
        if os.path.exists(fallback):
            return fallback
    return yaml_path


def rebuild_stsr(summaries: list, configs_root: str, task: str, field: str) -> dict:
    """Rebuild stsr_by_{field} from per_run.skill_results + YAML metadata (across
    episodes and runs).

    Episodes whose YAML is unavailable fall back to the prebuilt
    stsr_by_{field} in their episode_summary.
    """
    cnt, succ = defaultdict(int), defaultdict(int)
    fallback_list = []
    for ep_summary in summaries:
        ep_id = ep_summary["episode"]
        skill_meta = load_skill_meta(_episode_yaml(configs_root, task, ep_id))
        if not skill_meta:
            prebuilt = ep_summary.get(f"stsr_by_{field}")
            if prebuilt:
                fallback_list.append(prebuilt)
            continue
        for run_data in ep_summary.get("per_run", []):
            for r in run_data.get("skill_results", []):
                val = skill_meta.get(r["skill_id"], {}).get(field, f"<unknown_{field}>")
                cnt[val] += 1
                if r["success"]:
                    succ[val] += 1
    if cnt:
        return {
            t: {
                "n":         cnt[t],
                "n_success": succ[t],
                "SR":        round(succ[t] / cnt[t], 4) if cnt[t] else 0,
            }
            for t in sorted(cnt)
        }
    return merge_stsr(fallback_list)


def _load_merged_meta(summaries: list, configs_root: str, task: str) -> dict:
    """Merge skill metadata from all episode YAMLs (for matrix row labels).

    Different episodes may have different skill sets (e.g. later episodes add
    skills); merging one by one guarantees every skill_id gets a description
    label; for a duplicated skill_id the first YAML seen wins.
    """
    merged = {}
    for ep_summary in summaries:
        meta = load_skill_meta(_episode_yaml(configs_root, task, ep_summary["episode"]))
        for sid, info in (meta or {}).items():
            merged.setdefault(sid, info)
    return merged


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate episode summaries into task_summary.json"
    )
    parser.add_argument("--log_root", required=True,
                        help="task log dir (contains episode_*/episode_summary.json)")
    parser.add_argument("--task", required=True)
    parser.add_argument("--configs_root", default=None,
                        help="directory of episode YAMLs (default data/skill_eval_configs/{task})")
    args = parser.parse_args()

    configs_root = args.configs_root or str(
        PROJECT_ROOT / "data" / "skill_eval_configs" / args.task
    )

    ep_summary_paths = sorted(glob.glob(
        os.path.join(args.log_root, "episode_*", "episode_summary.json")
    ))
    if not ep_summary_paths:
        print(f"[WARN] No episode_summary.json found under {args.log_root}")
        return

    eps = []
    for p in ep_summary_paths:
        with open(p, encoding="utf-8") as f:
            eps.append(json.load(f))

    # ── 1. per_run aggregation: run i over all episodes ────────────────────
    max_runs = max((len(e.get("per_run", [])) for e in eps), default=0)
    per_run_summary = []

    for ri in range(1, max_runs + 1):
        run_tcrs, run_full, run_stsr_list = [], [], []

        for ep_summary in eps:
            run_data = next(
                (r for r in ep_summary.get("per_run", []) if r["run"] == ri),
                None
            )
            if run_data is None:
                continue
            run_tcrs.append(run_data["TSCR"])
            run_full.append(bool(run_data.get("full_success", False)))
            if "stsr_by_type" in run_data:
                run_stsr_list.append(run_data["stsr_by_type"])

        if not run_tcrs:
            continue

        entry = {
            "run":        ri,
            "n_episodes": len(run_tcrs),
            "mean_TSCR":   round(sum(run_tcrs) / len(run_tcrs), 4),
            "full_SR":    round(sum(run_full) / len(run_full), 4) if run_full else 0,
        }
        if run_stsr_list:
            entry["stsr_by_type"] = merge_stsr(run_stsr_list)
        per_run_summary.append(entry)

    # ── 2. Overall aggregation across runs and episodes ────────────────────────────────────────
    tcrs  = [e["TSCR"]["mean"]  for e in eps]
    ctcrs = [e["cTSCR"]["mean"] for e in eps]
    mean_tcr  = sum(tcrs)  / len(tcrs)
    mean_ctcr = sum(ctcrs) / len(ctcrs)
    std_tcr   = st.pstdev(tcrs)  if len(tcrs)  > 1 else 0.0
    std_ctcr  = st.pstdev(ctcrs) if len(ctcrs) > 1 else 0.0

    total_M    = sum(e["M"] for e in eps)
    total_succ = sum(e["TSCR"]["mean"] * e["M"] for e in eps)
    pooled_tcr = total_succ / total_M if total_M else 0.0

    full_srs    = [e.get("full_SR", 0) for e in eps]
    mean_full_sr = sum(full_srs) / len(full_srs) if full_srs else 0.0

    # ── 3. STSR: type / skill_description uniformly rebuilt from per_run + YAML ────────
    agg_stsr_by_type = rebuild_stsr(eps, configs_root, args.task, "type")
    stsr_by_desc     = rebuild_stsr(eps, configs_root, args.task, "skill_description")

    # ── 4. Write task_summary.json ───────────────────────────────────────────
    summary = {
        "task":       args.task,
        "n_episodes": len(eps),
        "TSCR": {
            "mean":   round(mean_tcr,  4),
            "std":    round(std_tcr,   4),
            "pooled": round(pooled_tcr, 4),
        },
        "cTSCR": {
            "mean": round(mean_ctcr, 4),
            "std":  round(std_ctcr,  4),
        },
        "full_SR":            round(mean_full_sr, 4),
        "STSR_by_type":       agg_stsr_by_type,
        "STSR_by_description": stsr_by_desc,
        "per_run":            per_run_summary,
        "per_episode": [
            {
                "episode":       e["episode"],
                "M":             e["M"],
                "TSCR":           e["TSCR"]["mean"],
                "cTSCR":          e["cTSCR"]["mean"],
                "full_SR":       e.get("full_SR", None),
                "FFI":           e.get("FFI_mode", e.get("FFI")),
                "stsr_by_type":  e.get("stsr_by_type", {}),
            }
            for e in eps
        ],
    }

    out_path = os.path.join(args.log_root, "task_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ── 5. Print summary ─────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print(f"Task: {summary['task']}  ({summary['n_episodes']} episodes)")
    print("-" * 70)
    print(f"  Mean-TSCR   : {summary['TSCR']['mean']:.4f} ± {summary['TSCR']['std']:.4f}")
    print(f"  Pooled-TSCR : {summary['TSCR']['pooled']:.4f}")
    print(f"  Mean-cTSCR  : {summary['cTSCR']['mean']:.4f} ± {summary['cTSCR']['std']:.4f}")
    print(f"  full_SR    : {summary['full_SR']:.4f}  (mean fraction of runs with all skills successful)")
    print("-" * 70)
    if per_run_summary:
        print("  Per-run cross-episode (run i aggregated over all trajectories):")
        for r in per_run_summary:
            stsr_str = ""
            if "stsr_by_type" in r:
                parts = [f"{t}={v['SR']*100:.0f}%" for t, v in r["stsr_by_type"].items()]
                stsr_str = "  [" + ", ".join(parts) + "]"
            print(f"    run_{r['run']:02d}: TSCR={r['mean_TSCR']:.4f}  "
                  f"full_SR={r['full_SR']:.4f}{stsr_str}")
        print("-" * 70)
    if agg_stsr_by_type:
        print("  STSR by type (aggregate):")
        for t, v in agg_stsr_by_type.items():
            print(f"    {t:<26} {v['n_success']:>3}/{v['n']:<3}  {v['SR']*100:>5.1f}%")

    # ── 6. Per-skill cross-episode matrix (rows = skill, columns = episode) ─────────────────
    skill_meta = _load_merged_meta(eps, configs_root, args.task)
    skill_ids = sorted({p["skill_id"] for e in eps for p in e.get("per_skill_SR", [])})
    if skill_ids:
        print("-" * 70)
        print("  Per-skill cross-episode matrix:")
        header = ["skill"] + [e["episode"] for e in eps] + ["mean"]
        print("    | " + " | ".join(header) + " |")
        for sid in skill_ids:
            label = skill_meta.get(sid, {}).get(
                "skill_description", f"skill_{sid}"
            )
            row, tot_s, tot_n = [str(label)], 0, 0
            for e in eps:
                p = next((x for x in e.get("per_skill_SR", []) if x["skill_id"] == sid), None)
                if p and p["n_runs"]:
                    row.append(f"{p['n_success']}/{p['n_runs']}")
                    tot_s += p["n_success"]
                    tot_n += p["n_runs"]
                else:
                    row.append("-")
            row.append(f"{tot_s}/{tot_n}" if tot_n else "-")
            print("    | " + " | ".join(row) + " |")

    print("=" * 70)
    print(f"  → {out_path}")


if __name__ == "__main__":
    main()
