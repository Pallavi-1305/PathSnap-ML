"""GroupKFold comparison of the five classifiers with full-observation misses."""
from __future__ import annotations
import argparse
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pathsnap.models import MODEL_FAMILY, check_columns, cross_validate, fit_and_score_holdout
from pathsnap.probabilities import candidate_probability_export
from pathsnap.viz import plot_cv_macro_f1, plot_log_loss, plot_student_comparison

METRICS = ["accuracy", "precision", "recall", "f1", "macro_f1", "log_loss",
           "top1_acc", "observation_arc_accuracy", "candidate_miss_rate"]
DETAILS = {
    "logreg": ("LogisticRegression(C=1, max_iter=200)", "binary log loss with L2 penalty",
               "Fast, calibrated probability baseline; linear boundary can underfit interactions."),
    "naive_bayes": ("GaussianNB(var_smoothing=1e-9)", "Gaussian class-conditional likelihood",
                    "Fast generative baseline; conditional feature-independence assumption is restrictive."),
    "decision_tree": ("DecisionTree(max_depth=12, min_samples_leaf=50)", "Gini impurity",
                      "Captures nonlinear rules; a single tree can be unstable and underfit at strict limits."),
    "random_forest": ("RandomForest(n_estimators=50, max_depth=12, max_samples=0.2)", "Gini impurity, bagged trees",
                      "Robust nonlinear model; larger inference and memory cost, probabilities can be uncalibrated."),
    "hist_gradient_boosting": ("HistGradientBoosting(learning_rate=0.1, max_iter=200)", "Binomial log loss",
                               "Efficient nonlinear boosting; can overfit if iterations/depth are too large."),
}


def load_candidates(path: Path, feats: list[str]) -> pd.DataFrame:
    cols = ["trip_id", "gps_point_id", "candidate_arc_id", "label"] + feats
    df = pd.read_csv(path, usecols=cols,
                     dtype={"trip_id": str, "gps_point_id": "int64", "candidate_arc_id": "int64"})
    df["trip_id"] = df.trip_id.astype(str)
    return df


def save_confusion(name: str, metrics: dict, out: Path) -> None:
    cm = np.array([[metrics["confusion_tn"], metrics["confusion_fp"]],
                   [metrics["confusion_fn"], metrics["confusion_tp"]]], dtype=int)
    pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"]).to_csv(out / f"{name}.csv")
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    image = ax.imshow(cm, cmap="Blues")
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, f"{value:,}", ha="center", va="center", color="black")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Predicted 0", "Predicted 1"],
           yticklabels=["True 0", "True 1"], title=f"{name} student-test confusion matrix")
    fig.colorbar(image, ax=ax); fig.tight_layout(); fig.savefig(out / f"{name}.png", dpi=150); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--folds", type=int)
    parser.add_argument("--candidate-file", type=Path)
    parser.add_argument("--observation-file", type=Path)
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    mcfg = dict(cfg["modelling"], seed=cfg["seed"])
    if args.folds:
        mcfg["cv_folds"] = args.folds
    feats = cfg["features"]
    names = args.models or mcfg["models"]
    candidate_path = args.candidate_file or ROOT / cfg["data"]["candidates_file"]
    observation_path = args.observation_file or candidate_path.parent / "observation_accounting.csv"
    if not candidate_path.exists():
        raise FileNotFoundError(f"Run run_preprocessing.py first; candidate table missing: {candidate_path}")
    if not observation_path.exists():
        raise FileNotFoundError(f"Full GPS observation ledger missing: {observation_path}")
    splits = pd.read_csv(ROOT / cfg["data"]["splits_file"], dtype={"trip_id": str})
    cand = load_candidates(candidate_path, feats)
    check_columns(cand, feats, "candidate table")
    obs = pd.read_csv(observation_path, dtype={"trip_id": str})
    if not {"trip_id", "gps_point_id", "has_correct_candidate"}.issubset(obs.columns):
        raise ValueError("Observation accounting file lacks required full-point fields")
    unknown = set(cand.trip_id) - set(splits.trip_id.astype(str))
    if unknown:
        raise ValueError(f"Candidate trips absent from splits.csv: {sorted(unknown)[:5]}")
    train_ids = set(splits.loc[splits.partition == "train", "trip_id"].astype(str))
    test_ids = set(splits.loc[splits.partition == "student_test", "trip_id"].astype(str))
    train = cand[cand.trip_id.isin(train_ids)].reset_index(drop=True)
    test = cand[cand.trip_id.isin(test_ids)].reset_index(drop=True)
    train_obs = obs[obs.trip_id.isin(train_ids)].reset_index(drop=True)
    test_obs = obs[obs.trip_id.isin(test_ids)].reset_index(drop=True)
    check_columns(train, feats, "training trips")
    check_columns(test, feats, "student-test trips")
    if set(train.trip_id) & set(test.trip_id):
        raise RuntimeError("Train/test trip leakage in candidate table")
    print(f"Train: {train.trip_id.nunique()} trips, {len(train):,} candidates, {len(train_obs):,} original observations")
    print(f"Student test: {test.trip_id.nunique()} trips, {len(test):,} candidates, {len(test_obs):,} original observations")

    out = ROOT / "results"; figures = out / "figures"
    matrices = out / "confusion_matrices"
    for directory in (out, figures, matrices): directory.mkdir(parents=True, exist_ok=True)
    cv_all, summary, prediction_frames, class_rows = [], [], [], []
    for name in names:
        started = time.perf_counter()
        folds = cross_validate(name, train, feats, mcfg, observations=train_obs)
        metrics, preds, fit_s, infer_us, train_metrics = fit_and_score_holdout(
            name, train, test, feats, mcfg, train_observations=train_obs,
            test_observations=test_obs)
        folds["model"] = name; cv_all.append(folds); prediction_frames.append(preds)
        row = {"model": name, "family": MODEL_FAMILY[name],
               "cv_fit_time_s_mean": float(folds.fit_time_s.mean()),
               "student_fit_time_s": fit_s, "student_infer_us_per_candidate": infer_us,
               "train_observation_count": len(train_obs), "student_observation_count": len(test_obs),
               "route_continuity": "N/A (evaluated during HMM stage)"}
        for metric in METRICS:
            row[f"cv_{metric}_mean"] = float(folds[metric].mean())
            row[f"cv_{metric}_std"] = float(folds[metric].std())
            row[f"train_{metric}"] = float(train_metrics[metric])
            row[f"student_{metric}"] = float(metrics[metric])
        for key in ("confusion_tn", "confusion_fp", "confusion_fn", "confusion_tp"):
            row[f"student_{key}"] = metrics[key]
        for class_id in (0, 1):
            for metric in ("precision", "recall", "f1"):
                row[f"student_class_{class_id}_{metric}"] = metrics[f"class_{class_id}_{metric}"]
                class_rows.append({"model": name, "partition": "student_test", "class": class_id,
                                   "metric": metric, "value": metrics[f"class_{class_id}_{metric}"]})
        summary.append(row)
        save_confusion(name, metrics, matrices)
        print(f"{name}: CV macro-F1={folds.macro_f1.mean():.4f}; student macro-F1={metrics['macro_f1']:.4f}; "
              f"student top-1={metrics['top1_acc']:.4f}; {time.perf_counter()-started:.1f}s")

    summary_df = pd.DataFrame(summary)
    cv_df = pd.concat(cv_all, ignore_index=True)
    # Selection uses training-only CV macro F1. Student scores never choose the model.
    best = str(summary_df.sort_values(["cv_macro_f1_mean", "model"], ascending=[False, True]).iloc[0].model)
    prediction_df = pd.concat(prediction_frames, ignore_index=True)
    (out / "cv_fold_results.csv").write_text(cv_df.to_csv(index=False), encoding="utf-8")
    summary_df.to_csv(out / "model_comparison.csv", index=False)
    prediction_df.to_csv(out / "student_test_predictions.csv", index=False)
    class_df = pd.DataFrame(class_rows)
    class_df.to_csv(out / "model_classification_report.csv", index=False)
    selected_rows = prediction_df[prediction_df.model_name == best]
    selected_prob = candidate_probability_export(
        selected_rows, selected_rows.predicted_probability.to_numpy(), best)
    selected_prob.to_csv(out / "candidate_probabilities.csv", index=False)
    # Exact identifiers are retained; probability export deliberately excludes labels.
    details = []
    for name, (params, objective, notes) in DETAILS.items():
        details.append({"model": name, "family": MODEL_FAMILY[name], "default_hyperparameters": params,
                        "objective_or_loss": objective, "strengths_and_limitations": notes})
    pd.DataFrame(details).to_csv(out / "model_details.csv", index=False)
    (out / "selected_model.json").write_text(json.dumps({
        "selected_model": best, "selection_metric": "mean training GroupKFold macro F1",
        "cv_macro_f1": float(summary_df.loc[summary_df.model == best, "cv_macro_f1_mean"].iloc[0])}, indent=2), encoding="utf-8")
    plot_cv_macro_f1(cv_df, figures / "cv_macro_f1.png")
    plot_student_comparison(summary_df, figures / "student_comparison.png")
    plot_log_loss(summary_df, figures / "log_loss.png")
    print(f"Selected by training-only CV: {best}")
    print("Saved model metrics, confusion matrices, all-model predictions, and the label-free selected probability export.")


if __name__ == "__main__":
    main()
