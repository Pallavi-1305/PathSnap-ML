"""Score the previous 50 m / 10 candidate configuration on training folds.

The large radius experiment already contains this candidate superset. A strict
positive is retained only when that exact positive remains after the baseline
radius/cap filter, so this does not relax labeling or access student labels.
"""
from pathlib import Path
import sys
import yaml
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pathsnap.models import cross_validate
from pathsnap.pipeline import account_candidates
from preprocessing.candidate_generation import add_relative_features

def main():
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    exp = ROOT / "data/processed/radius_experiments"
    source = pd.read_csv(exp / "r200_c30_candidates.csv", dtype={"trip_id": str})
    observations = pd.read_csv(exp / "r200_c30_observations.csv", dtype={"trip_id": str})
    # The existing positive identifies the closest correct arc in the full
    # superset. Keep it only if it survives the smaller radius and cap.
    base = source.loc[source.distance_m.le(50)].copy()
    base = base.sort_values(["trip_id", "gps_point_id", "distance_m", "candidate_arc_id"])
    base = base.groupby(["trip_id", "gps_point_id"], sort=False).head(10).copy()
    positive_ids = source.loc[source.label.eq(1), ["trip_id", "gps_point_id", "candidate_arc_id"]]
    base = base.drop(columns="label").merge(
        positive_ids.assign(label=1), on=["trip_id", "gps_point_id", "candidate_arc_id"], how="left")
    base["label"] = base.label.fillna(0).astype(int)
    base = add_relative_features(base)
    obs = account_candidates(observations, base)
    name = cfg["matching"].get("radius_selection_model", "random_forest")
    cv = cross_validate(name, base, cfg["features"],
                        dict(cfg["modelling"], seed=cfg["seed"]), observations=obs)
    cv["radius_m"], cv["max_candidates"] = 50, 10
    cv.to_csv(ROOT / "results/candidate_radius_baseline_folds.csv", index=False)
    fold_coverage = []
    splitter = GroupKFold(n_splits=cfg["modelling"]["cv_folds"])
    for fold, (_, val_idx) in enumerate(splitter.split(base, groups=base.trip_id)):
        val_ids = set(base.iloc[val_idx].trip_id.astype(str))
        val_obs = obs[obs.trip_id.astype(str).isin(val_ids)]
        fold_coverage.append({"fold": fold,
            "candidate_recall": float(val_obs.has_correct_candidate.mean()),
            "candidate_miss_rate": float(val_obs.candidate_miss.mean()),
            "mean_candidates_per_original_observation": float(val_obs.candidate_count.mean())})
    coverage = pd.DataFrame(fold_coverage)
    row = {"radius_m":50,"max_candidates":10,"model":name,
           "cv_macro_f1_mean":float(cv.macro_f1.mean()),
           "cv_macro_f1_std":float(cv.macro_f1.std(ddof=1)),
           "cv_candidate_recall_mean":float(coverage.candidate_recall.mean()),
           "cv_candidate_miss_rate_mean":float(coverage.candidate_miss_rate.mean()),
           "pooled_candidate_recall":float(obs.has_correct_candidate.mean()),
           "pooled_candidate_miss_rate":float(obs.candidate_miss.mean()),
           "candidate_rows":len(base),"original_observations":len(obs),
           "mean_candidates_per_observation":float(obs.candidate_count.mean()),
           "candidate_count_p95":float(obs.candidate_count.quantile(.95)),
           "zero_candidate_observation_rate":float((~obs.has_candidate).mean()),
           "cv_model_fit_seconds_mean":float(cv.fit_time_s.mean()),
           "cv_model_infer_us_per_row_mean":float(cv.infer_us_per_row.mean())}
    base.to_csv(exp / "r50_c10_candidates.csv", index=False)
    obs.to_csv(exp / "r50_c10_observations.csv", index=False)
    pd.DataFrame([row]).to_csv(ROOT / "results/candidate_radius_baseline.csv", index=False)
    comparison_path = ROOT / "results/candidate_radius_comparison.csv"
    if comparison_path.exists():
        comparison = pd.read_csv(comparison_path)
        comparison = comparison.loc[comparison.radius_m != 50]
        comparison = pd.concat([comparison, pd.DataFrame([row])], ignore_index=True, sort=False)
        comparison.sort_values("radius_m").to_csv(comparison_path, index=False)
    print(row)

if __name__ == "__main__": main()
