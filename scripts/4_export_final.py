"""Refit the tuned CV-selected model, export it, then score student test once."""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pathsnap.models import check_columns, evaluate, predict_timed, subsample_train, top1_decision
from pathsnap.probabilities import candidate_probability_export


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    results = ROOT / "results"; artifacts = ROOT / "artifacts"
    selected = json.loads((results / "selected_model.json").read_text(encoding="utf-8"))
    tuned = json.loads((results / "best_params.json").read_text(encoding="utf-8"))
    model_name, params = selected["selected_model"], tuned["best_params"]
    if tuned["selected_model"] != model_name:
        raise ValueError("Best Optuna parameters do not match the CV-selected model")
    candidate_path = ROOT / cfg["data"]["candidates_file"]
    observation_path = candidate_path.parent / "observation_accounting.csv"
    candidates = pd.read_csv(candidate_path, dtype={"trip_id": str})
    observations = pd.read_csv(observation_path, dtype={"trip_id": str})
    splits = pd.read_csv(ROOT / cfg["data"]["splits_file"], dtype={"trip_id": str})
    train_ids = set(splits.loc[splits.partition == "train", "trip_id"].astype(str))
    student_ids = set(splits.loc[splits.partition == "student_test", "trip_id"].astype(str))
    train = candidates[candidates.trip_id.isin(train_ids)].reset_index(drop=True)
    student = candidates[candidates.trip_id.isin(student_ids)].reset_index(drop=True)
    train_obs = observations[observations.trip_id.isin(train_ids)].reset_index(drop=True)
    student_obs = observations[observations.trip_id.isin(student_ids)].reset_index(drop=True)
    feats = cfg["features"]
    check_columns(train, feats, "final training data")
    if set(train.trip_id) & set(student.trip_id):
        raise ValueError("Student-test trips leaked into final training data")
    fit_data = subsample_train(train, cfg["modelling"]["neg_per_obs"], cfg["seed"])
    model = __import__("pathsnap.models", fromlist=["build_model"]).build_model(
        model_name, cfg["seed"], params=params)
    t0 = time.perf_counter()
    model.fit(fit_data[feats].to_numpy(dtype=np.float32), fit_data.label.astype(int).to_numpy())
    train_seconds = time.perf_counter() - t0
    # One student-test evaluation after the training-only search has finished.
    probability, inference_seconds = predict_timed(model, student, feats)
    metrics = evaluate(student, probability, student_obs)
    train_probability, _ = predict_timed(model, train, feats)
    train_metrics = evaluate(train, train_probability, train_obs)
    artifacts.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifacts / "final_model.joblib")
    radius_selection_path = results / "candidate_radius_selection.json"
    radius_selection = json.loads(radius_selection_path.read_text(encoding="utf-8")) if radius_selection_path.exists() else {}
    metadata = {
        "model": model_name, "hyperparameters": params, "feature_columns": feats,
        "candidate_radius_m": radius_selection.get("radius_m", cfg.get("matching", {}).get("candidate_radius_m")),
        "candidate_cap": radius_selection.get("max_candidates", cfg.get("matching", {}).get("max_candidates")),
        "crs": cfg["data"]["crs"], "training_trip_count": len(train_ids),
        "tuning_trial": tuned["best_trial_number"],
        "training_cv_macro_f1": tuned["best_cv_macro_f1"],
        "hmm": cfg["hmm"],
    }
    (artifacts / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    probability_export = candidate_probability_export(student, probability, model_name)
    # This file intentionally contains no reference label or route information.
    probability_export.to_csv(results / "candidate_probabilities.csv", index=False)
    pd.DataFrame([{"partition": "train", **train_metrics},
                  {"partition": "student_test", **metrics}]).to_csv(results / "tuned_model_metrics.csv", index=False)
    baseline = pd.read_csv(results / "model_comparison.csv")
    base_row = baseline.loc[baseline.model == model_name].iloc[0]
    compare_rows = []
    compare_metrics = ["accuracy", "precision", "recall", "f1", "macro_f1", "log_loss",
                       "top1_acc", "observation_arc_accuracy", "candidate_miss_rate"]
    for key in compare_metrics:
        compare_rows.append({"metric": key, "untuned": float(base_row[f"student_{key}"]),
                             "tuned": float(metrics[key]),
                             "delta_tuned_minus_untuned": float(metrics[key] - base_row[f"student_{key}"])})
    pd.DataFrame(compare_rows).to_csv(results / "tuned_vs_untuned.csv", index=False)
    (results / "tuned_student_summary.json").write_text(json.dumps({
        "selected_model": model_name, "student_test_scored_once": True,
        "training_seconds": train_seconds,
        "inference_us_per_candidate": 1e6 * inference_seconds / max(len(student), 1),
        "metrics": metrics, "train_diagnostic_metrics": train_metrics,
    }, indent=2), encoding="utf-8")
    print(f"Saved frozen model: {artifacts / 'final_model.joblib'}")
    print(f"Exported {len(probability_export):,} label-free candidate probabilities for {student.trip_id.nunique()} student trips")
    print(f"Tuned-vs-untuned metrics: {results / 'tuned_vs_untuned.csv'}")
    print(f"One student-test evaluation complete; macro F1={metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
