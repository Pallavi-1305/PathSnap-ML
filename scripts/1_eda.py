"""Training-only candidate-radius experiments and data/network EDA.

Candidate geometry is generated once at the largest tested radius and then
filtered to each smaller radius. No student-test labels or evaluator data enter
the radius or cap selection.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pathsnap.models import cross_validate
from pathsnap.pipeline import (account_candidates, filter_candidate_radius,
                               generate_trip_tables, resolve_raw_root, trip_folder)


def weak_components(nodes: pd.DataFrame, arcs: pd.DataFrame) -> tuple[int, int]:
    ids = nodes.node_id.astype(int).tolist()
    parent = {node: node for node in ids}
    degree = {node: 0 for node in ids}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    valid = arcs[arcs.from_node_id.isin(parent) & arcs.to_node_id.isin(parent)]
    for u, v in valid[["from_node_id", "to_node_id"]].itertuples(index=False, name=None):
        u, v = int(u), int(v)
        degree[u] += 1; degree[v] += 1
        union(u, v)
    return len({find(x) for x in ids}), sum(v == 0 for v in degree.values())


def append_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, mode="a", index=False, header=not path.exists())


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    dcfg, mcfg = cfg["data"], cfg["modelling"]
    match = cfg.setdefault("matching", {})
    radii = sorted(set(map(float, match.get("radius_experiments_m", [150, 175, 200]))))
    cap = int(match.get("max_candidates", 30))
    if not radii or min(radii) <= 0 or cap <= 0:
        raise SystemExit("Candidate radii and max_candidates must be positive")
    max_radius = max(radii)
    crs = dcfg.get("crs", "EPSG:32640")
    splits = pd.read_csv(ROOT / dcfg["splits_file"], dtype={"trip_id": str})
    train_ids = splits.loc[splits.partition == "train", "trip_id"].astype(str).tolist()
    if len(train_ids) != 80:
        print(f"Note: splits.csv contains {len(train_ids)} training IDs (brief expected 80).")
    raw_root = resolve_raw_root(ROOT, dcfg["raw_root"])
    exp_dir = ROOT / "data/processed/radius_experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)
    out_dir = ROOT / "results/eda"
    out_dir.mkdir(parents=True, exist_ok=True)
    for radius in radii:
        for suffix in ("candidates", "observations"):
            p = exp_dir / f"r{int(radius)}_c{cap}_{suffix}.csv"
            if p.exists():
                p.unlink()
    gps_rows, road_rows = [], []
    generation_seconds = 0.0
    radii_filter_seconds = {r: 0.0 for r in radii}
    for i, trip_id in enumerate(train_ids, 1):
        folder = trip_folder(raw_root, "train", trip_id)
        candidates, obs, trip, network, elapsed = generate_trip_tables(
            folder, trip_id, "train", radius_m=max_radius, max_candidates=cap,
            crs=crs, include_reference=True)
        generation_seconds += elapsed
        track = trip["track"]
        gaps = track.time_gap_s.dropna()
        distances = track.distance_m.dropna()
        gps_rows.append({
            "trip_id": trip_id, "gps_observations": len(track),
            "missing_coordinate_rows": int(track[["lon", "lat"]].isna().any(axis=1).sum()),
            "invalid_coordinate_rows": int((~track.coord_valid).sum()),
            "timestamp_gap_count": int(gaps.size),
            "timestamp_gap_median_s": float(gaps.median()) if len(gaps) else np.nan,
            "timestamp_gap_p95_s": float(gaps.quantile(.95)) if len(gaps) else np.nan,
            "nonpositive_timestamp_gaps": int((gaps <= 0).sum()),
            "trip_length_m": float(distances.sum()),
            "distance_step_median": float(distances.median()) if len(distances) else np.nan,
            "distance_step_p95": float(distances.quantile(.95)) if len(distances) else np.nan,
            "duplicate_observations": int(track.is_duplicate.sum()),
            "implausible_speed_observations": int(track.implausible_speed.sum()),
        })
        arcs, nodes = trip["arcs"], trip["nodes"]
        components, isolated = weak_components(nodes, arcs)
        outdegree = arcs.groupby("from_node_id").size()
        indegree = arcs.groupby("to_node_id").size()
        road_rows.append({
            "trip_id": trip_id, "nodes": len(nodes), "directed_arcs": len(arcs),
            "invalid_arcs": int((~network["arcs"].valid).sum()),
            "weak_components": components, "isolated_nodes": isolated,
            "mean_out_degree_nonisolated": float(outdegree.mean()) if len(outdegree) else 0.0,
            "mean_in_degree_nonisolated": float(indegree.mean()) if len(indegree) else 0.0,
        })
        # Base candidates are the max-radius, max-cap superset. Each radius
        # receives fresh strict labels after filtering and cap application.
        for radius in radii:
            t0 = time.perf_counter()
            selected = filter_candidate_radius(candidates, radius, cap, trip["route"])
            selected_obs = account_candidates(obs, selected)
            radii_filter_seconds[radius] += time.perf_counter() - t0
            append_csv(selected, exp_dir / f"r{int(radius)}_c{cap}_candidates.csv")
            append_csv(selected_obs, exp_dir / f"r{int(radius)}_c{cap}_observations.csv")
        if i % 10 == 0 or i == len(train_ids):
            print(f"Generated max-radius candidates for {i}/{len(train_ids)} training trips")

    gps = pd.DataFrame(gps_rows); road = pd.DataFrame(road_rows)
    gps.to_csv(out_dir / "gps_trip_summary.csv", index=False)
    road.to_csv(out_dir / "road_network_summary.csv", index=False)
    # Evaluate each radius with the existing group-aware model CV and per-point ledger.
    feats = cfg["features"]
    model_name = cfg.get("matching", {}).get("radius_selection_model", "random_forest")
    fold_rows, summary_rows = [], []
    for radius in radii:
        cp = exp_dir / f"r{int(radius)}_c{cap}_candidates.csv"
        op = exp_dir / f"r{int(radius)}_c{cap}_observations.csv"
        cand = pd.read_csv(cp, dtype={"trip_id": str})
        obs = pd.read_csv(op, dtype={"trip_id": str})
        for f in feats:
            if f not in cand:
                raise ValueError(f"Configured model feature {f!r} absent from candidate schema")
        cand["label"] = cand.label.astype(int)
        cv_cfg = dict(mcfg, seed=cfg["seed"])
        cv = cross_validate(model_name, cand, feats, cv_cfg, observations=obs)
        cv["radius_m"] = radius; cv["max_candidates"] = cap
        fold_rows.append(cv)
        splitter = GroupKFold(n_splits=mcfg["cv_folds"])
        fold_cov = []
        for fold, (_, val_idx) in enumerate(splitter.split(cand, groups=cand.trip_id)):
            val_ids = set(cand.iloc[val_idx].trip_id)
            val_obs = obs[obs.trip_id.isin(val_ids)]
            val_cand = cand[cand.trip_id.isin(val_ids)]
            fold_cov.append({
                "fold": fold,
                "candidate_recall": float(val_obs.has_correct_candidate.mean()) if len(val_obs) else 0.0,
                "candidate_miss_rate": float(val_obs.candidate_miss.mean()) if len(val_obs) else 1.0,
                "mean_candidates_per_original_observation": float(val_obs.candidate_count.mean()) if len(val_obs) else 0.0,
                "candidate_rows": len(val_cand),
            })
        cov = pd.DataFrame(fold_cov)
        nearest = obs.nearest_reference_distance_m.dropna()
        summary_rows.append({
            "radius_m": radius, "max_candidates": cap, "model": model_name,
            "cv_macro_f1_mean": float(cv.macro_f1.mean()),
            "cv_macro_f1_std": float(cv.macro_f1.std(ddof=1)),
            "cv_accuracy_mean": float(cv.accuracy.mean()),
            "cv_candidate_recall_mean": float(cov.candidate_recall.mean()),
            "cv_candidate_miss_rate_mean": float(cov.candidate_miss_rate.mean()),
            "candidate_rows": len(cand), "original_observations": len(obs),
            "mean_candidates_per_observation": float(obs.candidate_count.mean()),
            "candidate_count_p95": float(obs.candidate_count.quantile(.95)),
            "zero_candidate_observation_rate": float((~obs.has_candidate).mean()),
            "nearest_correct_distance_p50_m": float(nearest.quantile(.50)) if len(nearest) else np.nan,
            "nearest_correct_distance_p75_m": float(nearest.quantile(.75)) if len(nearest) else np.nan,
            "nearest_correct_distance_p95_m": float(nearest.quantile(.95)) if len(nearest) else np.nan,
            "max_radius_generation_seconds_shared": generation_seconds,
            "radius_filter_label_seconds": radii_filter_seconds[radius],
            "cv_model_fit_seconds_mean": float(cv.fit_time_s.mean()),
            "cv_model_infer_us_per_row_mean": float(cv.infer_us_per_row.mean()),
        })
        print(f"Radius {radius:g} m: CV macro F1={cv.macro_f1.mean():.4f}; "
              f"candidate recall={cov.candidate_recall.mean():.4f}; "
              f"mean candidates={obs.candidate_count.mean():.2f}")

    summary = pd.DataFrame(summary_rows)
    folds = pd.concat(fold_rows, ignore_index=True)
    summary.to_csv(ROOT / "results/candidate_radius_comparison.csv", index=False)
    folds.to_csv(ROOT / "results/candidate_radius_fold_results.csv", index=False)
    # Select the smallest candidate set within a configured CV F1 tolerance of
    # the best training-fold score; student-test scores are never consulted.
    tolerance = float(match.get("selection_f1_tolerance", 0.002))
    peak = float(summary.cv_macro_f1_mean.max())
    eligible = summary[summary.cv_macro_f1_mean >= peak - tolerance]
    chosen = eligible.sort_values(["mean_candidates_per_observation", "radius_m"]).iloc[0]
    selection = {"radius_m": float(chosen.radius_m), "max_candidates": cap,
                 "selection_model": model_name, "selection_metric": "training GroupKFold macro F1",
                 "f1_tolerance": tolerance, "selected_cv_macro_f1": float(chosen.cv_macro_f1_mean),
                 "selected_cv_candidate_recall": float(chosen.cv_candidate_recall_mean),
                 "selected_candidate_miss_rate": float(chosen.cv_candidate_miss_rate_mean)}
    (ROOT / "results/candidate_radius_selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")

    # EDA plots: GPS quality, candidate return/recall, nearest route distance, graph structure.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(gps.trip_length_m.dropna() / 1000, bins=20, color="#3b6ea5")
    ax.set(title="Training trip lengths", xlabel="Length (km)", ylabel="Trips")
    fig.tight_layout(); fig.savefig(out_dir / "trip_lengths.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(gps.distance_step_p95.dropna(), bins=25, color="#e08e45")
    ax.set(title="GPS step distance (per-trip 95th percentile)", xlabel="Distance (m)", ylabel="Trips")
    fig.tight_layout(); fig.savefig(out_dir / "gps_step_distances.png", dpi=150); plt.close(fig)
    fig, ax1 = plt.subplots(figsize=(8, 4.8))
    ax1.plot(summary.radius_m, summary.cv_candidate_recall_mean, marker="o", label="Candidate recall", color="#2ca02c")
    ax1.plot(summary.radius_m, summary.cv_candidate_miss_rate_mean, marker="o", label="Candidate miss rate", color="#d62728")
    ax1.set(xlabel="Search radius (m)", ylabel="Rate", ylim=(0, 1), title="Training-fold candidate recall and misses")
    ax2 = ax1.twinx(); ax2.plot(summary.radius_m, summary.mean_candidates_per_observation,
                               marker="s", label="Mean candidates", color="#1f77b4")
    ax2.set_ylabel("Mean candidate count per original GPS observation")
    lines = ax1.get_lines() + ax2.get_lines(); ax1.legend(lines, [line.get_label() for line in lines], loc="center right")
    fig.tight_layout(); fig.savefig(out_dir / "candidate_radius_tradeoff.png", dpi=150); plt.close(fig)
    nearest = pd.concat([pd.read_csv(exp_dir / f"r{int(radii[-1])}_c{cap}_observations.csv",
                                     usecols=["nearest_reference_distance_m"])
                         for _ in [0]], ignore_index=True).nearest_reference_distance_m.dropna()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(nearest.clip(upper=1000), bins=40, color="#3b6ea5")
    ax.set(title="Nearest reference-route arc distance (training observations)",
           xlabel="Distance (m; values clipped at 1,000 m for display)", ylabel="Observations")
    fig.tight_layout(); fig.savefig(out_dir / "nearest_correct_distance.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(road.weak_components, bins=20, color="#9467bd")
    ax.set(title="Weakly connected components by training road network",
           xlabel="Components", ylabel="Trips")
    fig.tight_layout(); fig.savefig(out_dir / "road_components.png", dpi=150); plt.close(fig)

    print(f"Selected radius={selection['radius_m']:g} m, cap={cap} using training-only CV; saved {ROOT / 'results/candidate_radius_selection.json'}")
    print(f"EDA tables and plots: {out_dir}")


if __name__ == "__main__":
    main()
