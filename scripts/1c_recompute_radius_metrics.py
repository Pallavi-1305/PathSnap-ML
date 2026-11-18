"""Recompute radius metrics from saved candidate shards after ledger fixes."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pathsnap.models import cross_validate
from pathsnap.pipeline import account_candidates

def main():
    cfg=yaml.safe_load((ROOT/"config.yaml").read_text(encoding="utf-8"))
    exp=ROOT/"data/processed/radius_experiments"
    summaries=[]; folds=[]
    for radius in (150,175,200):
        cp=exp/f"r{radius}_c30_candidates.csv"; op=exp/f"r{radius}_c30_observations.csv"
        cand=pd.read_csv(cp,dtype={"trip_id":str})
        original=pd.read_csv(op,dtype={"trip_id":str})
        obs=account_candidates(original,cand)
        obs.to_csv(op,index=False)
        cv=cross_validate("random_forest",cand,cfg["features"],
                          dict(cfg["modelling"],seed=cfg["seed"]),observations=obs)
        cv["radius_m"],cv["max_candidates"]=radius,30; folds.append(cv)
        nearest=obs.nearest_reference_distance_m.dropna()
        summaries.append({"radius_m":radius,"max_candidates":30,"model":"random_forest",
          "cv_macro_f1_mean":float(cv.macro_f1.mean()),"cv_macro_f1_std":float(cv.macro_f1.std(ddof=1)),
          "cv_accuracy_mean":float(cv.accuracy.mean()),
          "cv_candidate_recall_mean":float(obs.has_correct_candidate.mean()),
          "cv_candidate_miss_rate_mean":float(obs.candidate_miss.mean()),
          "candidate_rows":len(cand),"original_observations":len(obs),
          "mean_candidates_per_observation":float(obs.candidate_count.mean()),
          "candidate_count_p95":float(obs.candidate_count.quantile(.95)),
          "zero_candidate_observation_rate":float((~obs.has_candidate).mean()),
          "nearest_correct_distance_p50_m":float(nearest.quantile(.5)),
          "nearest_correct_distance_p75_m":float(nearest.quantile(.75)),
          "nearest_correct_distance_p95_m":float(nearest.quantile(.95)),
          "cv_model_fit_seconds_mean":float(cv.fit_time_s.mean()),
          "cv_model_infer_us_per_row_mean":float(cv.infer_us_per_row.mean())})
        print(f"{radius} m: recall={obs.has_correct_candidate.mean():.4f}, misses={obs.candidate_miss.mean():.4f}, CV macro F1={cv.macro_f1.mean():.4f}",flush=True)
    summary=pd.DataFrame(summaries); summary.to_csv(ROOT/"results/candidate_radius_comparison.csv",index=False)
    pd.concat(folds,ignore_index=True).to_csv(ROOT/"results/candidate_radius_fold_results.csv",index=False)
    tolerance=float(cfg["matching"].get("selection_f1_tolerance",.002)); peak=summary.cv_macro_f1_mean.max()
    eligible=summary[summary.cv_macro_f1_mean>=peak-tolerance]
    chosen=eligible.sort_values(["mean_candidates_per_observation","radius_m"]).iloc[0]
    import json
    selection={"radius_m":float(chosen.radius_m),"max_candidates":30,"selection_model":"random_forest",
      "selection_metric":"training GroupKFold macro F1","f1_tolerance":tolerance,
      "selected_cv_macro_f1":float(chosen.cv_macro_f1_mean),
      "selected_cv_candidate_recall":float(chosen.cv_candidate_recall_mean),
      "selected_candidate_miss_rate":float(chosen.cv_candidate_miss_rate_mean)}
    (ROOT/"results/candidate_radius_selection.json").write_text(json.dumps(selection,indent=2),encoding="utf-8")
if __name__=="__main__": main()
