"""Generate candidate features and a complete one-row-per-GPS observation ledger."""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from pathsnap.pipeline import generate_trip_tables, resolve_raw_root, trip_folder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radius", type=float, help="Override selected candidate radius")
    parser.add_argument("--maxcand", type=int, help="Override candidate cap")
    parser.add_argument("--raw-root", type=Path, help="Override raw dataset root")
    parser.add_argument("--train-only", action="store_true", help="Do not generate student-test features")
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    match, dcfg = cfg.get("matching", {}), cfg["data"]
    selection_path = ROOT / "results/candidate_radius_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8")) if selection_path.exists() else {}
    radius = args.radius or selection.get("radius_m", match.get("candidate_radius_m", 200))
    cap = args.maxcand or selection.get("max_candidates", match.get("max_candidates", 30))
    raw_root = args.raw_root or resolve_raw_root(ROOT, dcfg["raw_root"])
    splits = pd.read_csv(ROOT / dcfg["splits_file"], dtype={"trip_id": str})
    partitions = ["train"] if args.train_only else ["train", "student_test"]
    out_dir = ROOT / "data/processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = out_dir / "candidate_features.csv"
    observation_path = out_dir / "observation_accounting.csv"
    for path in (candidate_path, observation_path):
        path.unlink(missing_ok=True)
    total_candidates = total_observations = 0
    trip_summaries = []
    for partition in partitions:
        ids = splits.loc[splits.partition == partition, "trip_id"].astype(str).tolist()
        if not ids:
            raise SystemExit(f"No {partition} trip IDs in splits file")
        for i, trip_id in enumerate(ids, 1):
            start = time.perf_counter()
            candidates, observations, trip, _, elapsed = generate_trip_tables(
                trip_folder(raw_root, partition, trip_id), trip_id, partition,
                radius_m=float(radius), max_candidates=int(cap), crs=dcfg["crs"],
                include_reference=True)
            if not candidates.empty:
                candidates["partition"] = partition
                candidates.to_csv(candidate_path, mode="a", index=False,
                                  header=not candidate_path.exists())
            observations.to_csv(observation_path, mode="a", index=False,
                                header=not observation_path.exists())
            total_candidates += len(candidates)
            total_observations += len(observations)
            trip_summaries.append({
                "trip_id": trip_id, "partition": partition,
                "gps_observations": len(observations), "candidate_rows": len(candidates),
                "zero_candidate_observations": int((~observations.has_candidate).sum()),
                "candidate_miss_observations": int(observations.candidate_miss.sum()),
                "candidate_recall": float(observations.has_correct_candidate.mean()),
                "elapsed_seconds": time.perf_counter() - start,
            })
            if i % 10 == 0 or i == len(ids):
                print(f"{partition}: processed {i}/{len(ids)} trips")
    summary = pd.DataFrame(trip_summaries)
    (ROOT / "results").mkdir(exist_ok=True)
    summary.to_csv(ROOT / "results/preprocessing_trip_summary.csv", index=False)
    print(f"Candidate settings: radius={radius:g} m, cap={cap}")
    print(f"Saved {total_candidates:,} candidate rows and {total_observations:,} original observations")
    print(f"Candidates: {candidate_path}")
    print(f"Observation ledger: {observation_path}")
    print(f"Overall candidate recall: {summary.candidate_recall.mean():.4f} (trip mean; inspect partition rows separately)")


if __name__ == "__main__":
    main()
