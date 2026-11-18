"""Five candidate models, group-aware CV by trip, and metrics.

Input (Person 1): data/processed/candidate_features.csv with one row per (trip, GPS point, candidate arc):
    trip_id, gps_point_id, candidate_arc_id, <feature columns listed in config.yaml>, label
label = 1 if the candidate arc is on the reference route (set-membership labelling), else 0.

Imbalance handling: training uses subsampling of wrong candidates only (no class weights).
Decision rule for macro F1: the top-scoring candidate per GPS point is predicted 1, all others 0.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, log_loss, precision_score, recall_score)
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

OBS_COLS = ["trip_id", "gps_point_id"]
ID_COLS = ["trip_id", "gps_point_id", "candidate_arc_id", "label"]

MODEL_FAMILY = {
    "logreg": "Linear, probabilistic",
    "naive_bayes": "Probabilistic, generative (Gaussian Naive Bayes)",
    "decision_tree": "Tree-based (single decision tree)",
    "random_forest": "Bagged trees (ensemble)",
    "hist_gradient_boosting": "Boosting (histogram)",
}


def build_model(name: str, seed: int, params: dict | None = None):
    """Untrained pipeline: median imputation (fitted per training fold) -> scaling -> model."""
    if name == "logreg":
        est = LogisticRegression(max_iter=200, random_state=seed)
    elif name == "naive_bayes":
        est = GaussianNB()
    elif name == "decision_tree":
        est = DecisionTreeClassifier(max_depth=12, min_samples_leaf=50, random_state=seed)
    elif name == "random_forest":
        est = RandomForestClassifier(
            n_estimators=50, max_depth=12, max_samples=0.2, n_jobs=-1, random_state=seed
        )
    elif name == "hist_gradient_boosting":
        est = HistGradientBoostingClassifier(
            learning_rate=0.1, max_iter=200, early_stopping=True, random_state=seed
        )
    else:
        raise ValueError(f"Unknown model: {name}")
    if params:
        est.set_params(**params)
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), est)


def check_columns(df: pd.DataFrame, feats: list[str], name: str) -> None:
    missing = [c for c in feats + ID_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing columns {missing}")
    if df["label"].isna().any():
        raise ValueError(f"{name}: {int(df['label'].isna().sum())} rows have no label; "
                         "student-test trips must be labelled for scoring")


def subsample_train(df: pd.DataFrame, neg_per_obs: int, seed: int) -> pd.DataFrame:
    """Training only. Keep every correct candidate and up to neg_per_obs wrong ones per GPS point.
    If distance_m exists, the nearest wrong candidates are kept (hard negatives)."""
    pos = df[df["label"] == 1]
    neg = df[df["label"] == 0]
    if "distance_m" in neg.columns:
        neg = neg.sort_values(OBS_COLS + ["distance_m"])
    else:
        neg = neg.sample(frac=1.0, random_state=seed)
    neg = neg.groupby(OBS_COLS, sort=False).head(neg_per_obs)
    return pd.concat([pos, neg]).sample(frac=1.0, random_state=seed)


def top1_decision(df: pd.DataFrame, p) -> np.ndarray:
    """1 for the highest-scoring candidate of each GPS point, 0 for all others."""
    d = pd.DataFrame({
        "trip_id": df["trip_id"].to_numpy(),
        "gps_point_id": df["gps_point_id"].to_numpy(),
        "p": np.asarray(p, dtype=float),
    })
    idx = d.groupby(OBS_COLS, sort=False)["p"].idxmax().to_numpy()
    dec = np.zeros(len(d), dtype=int)
    dec[idx] = 1
    return dec


def candidate_metrics(y, p, dec, missed_observations: int = 0) -> dict:
    """Candidate-row classification metrics plus one accounting FN per missed point."""
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    dec = np.asarray(dec, dtype=int)
    if missed_observations:
        y = np.concatenate([y, np.ones(missed_observations, dtype=int)])
        p = np.concatenate([p, np.zeros(missed_observations, dtype=float)])
        dec = np.concatenate([dec, np.zeros(missed_observations, dtype=int)])
    p = np.clip(np.nan_to_num(p, nan=0.0, posinf=1.0, neginf=0.0), 1e-15, 1 - 1e-15)
    cm = confusion_matrix(y, dec, labels=[0, 1])
    report = classification_report(y, dec, labels=[0, 1], output_dict=True, zero_division=0)
    result = {
        "accuracy": float(accuracy_score(y, dec)),
        "precision": float(precision_score(y, dec, zero_division=0)),
        "recall": float(recall_score(y, dec, zero_division=0)),
        "f1": float(f1_score(y, dec, zero_division=0)),
        "macro_f1": float(f1_score(y, dec, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "confusion_tn": int(cm[0, 0]), "confusion_fp": int(cm[0, 1]),
        "confusion_fn": int(cm[1, 0]), "confusion_tp": int(cm[1, 1]),
    }
    for class_id in (0, 1):
        for short, full in (("precision", "precision"), ("recall", "recall"), ("f1", "f1-score")):
            result[f"class_{class_id}_{short}"] = float(report[str(class_id)][full])
    return result


def observation_metrics(df: pd.DataFrame, p, dec: np.ndarray,
                        observations: pd.DataFrame | None = None) -> dict:
    """Score every original observation; None predictions remain explicit misses."""
    d = pd.DataFrame({
        "trip_id": df["trip_id"].to_numpy(),
        "gps_point_id": df["gps_point_id"].to_numpy(),
        "label": df["label"].to_numpy().astype(int),
        "dec": dec,
    })
    top = d[d["dec"] == 1]
    has_correct = d.groupby(OBS_COLS)["label"].max()
    if observations is None:
        observations = (has_correct.rename("has_correct_candidate").reset_index()
                        .assign(candidate_count=1))
    n_obs = len(observations)
    missed = int((~observations["has_correct_candidate"].astype(bool)).sum())
    top_correct = int(top["label"].sum())
    pred = pd.DataFrame({"trip_id": df.trip_id.to_numpy(), "gps_point_id": df.gps_point_id.to_numpy(),
                         "candidate_arc_id": df.candidate_arc_id.to_numpy(), "probability": np.asarray(p)})
    if not pred.empty:
        pred = pred.loc[pred.groupby(OBS_COLS)["probability"].idxmax()]
    pred_map = {(str(r.trip_id), int(r.gps_point_id)): int(r.candidate_arc_id)
                for r in pred.itertuples(index=False)}
    arc_hits = 0
    known_reference = 0
    if "nearest_reference_arc_id" in observations:
        for r in observations.itertuples(index=False):
            truth = getattr(r, "nearest_reference_arc_id")
            if pd.notna(truth) and int(truth) >= 0:
                known_reference += 1
                arc_hits += pred_map.get((str(r.trip_id), int(r.gps_point_id)), -1) == int(truth)
    else:
        arc_hits = top_correct
        known_reference = n_obs
    return {
        "top1_acc": float(top_correct / n_obs) if n_obs else 0.0,
        "observation_arc_accuracy": float(arc_hits / n_obs) if n_obs else 0.0,
        "known_reference_observations": known_reference,
        "observation_count": n_obs,
        "candidate_miss_rate": float(missed / n_obs) if n_obs else 0.0,
    }


def fit_model(name: str, df: pd.DataFrame, feats: list[str], seed: int,
              params: dict | None = None):
    model = build_model(name, seed, params=params)
    X = df[feats].to_numpy(dtype=np.float32)
    t0 = time.perf_counter()
    model.fit(X, df["label"].to_numpy().astype(int))
    return model, time.perf_counter() - t0


def predict_timed(model, df: pd.DataFrame, feats: list[str]):
    X = df[feats].to_numpy(dtype=np.float32)
    t0 = time.perf_counter()
    p = model.predict_proba(X)[:, 1]
    return p, time.perf_counter() - t0


def evaluate(df: pd.DataFrame, p, observations: pd.DataFrame | None = None) -> dict:
    dec = top1_decision(df, p)
    missed = int((~observations["has_correct_candidate"].astype(bool)).sum()) if observations is not None else 0
    return {**candidate_metrics(df["label"], p, dec, missed),
            **observation_metrics(df, p, dec, observations)}


def cross_validate(name: str, train_df: pd.DataFrame, feats: list[str], cfg: dict,
                   observations: pd.DataFrame | None = None, params: dict | None = None):
    """GroupKFold by trip on training trips. Imputer and scaler are fitted inside each training fold."""
    gkf = GroupKFold(n_splits=cfg["cv_folds"])
    rows = []
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_df, groups=train_df["trip_id"])):
        tr, va = train_df.iloc[tr_idx], train_df.iloc[va_idx]
        if set(tr["trip_id"]) & set(va["trip_id"]):
            raise RuntimeError("Trip appears in both train and validation of one fold")
        fit_df = subsample_train(tr, cfg["neg_per_obs"], cfg["seed"])
        model, fit_s = fit_model(name, fit_df, feats, cfg["seed"], params=params)
        p, inf_s = predict_timed(model, va, feats)  # validation uses ALL candidates
        row = {
            "model": name, "fold": fold,
            "n_fit_rows": len(fit_df), "n_val_rows": len(va),
            "fit_time_s": fit_s, "infer_us_per_row": 1e6 * inf_s / len(va),
        }
        fold_obs = observations[observations.trip_id.isin(set(va.trip_id))] if observations is not None else None
        row.update(evaluate(va, p, fold_obs))
        rows.append(row)
    return pd.DataFrame(rows)


def fit_and_score_holdout(name: str, train_df, test_df, feats, cfg: dict,
                          train_observations: pd.DataFrame | None = None,
                          test_observations: pd.DataFrame | None = None,
                          params: dict | None = None):
    """Fit on all training trips (subsampled); score student-test trips once (all candidates)."""
    fit_df = subsample_train(train_df, cfg["neg_per_obs"], cfg["seed"])
    model, fit_s = fit_model(name, fit_df, feats, cfg["seed"], params=params)
    p, inf_s = predict_timed(model, test_df, feats)
    metrics = evaluate(test_df, p, test_observations)
    train_p, _ = predict_timed(model, train_df, feats)
    train_metrics = evaluate(train_df, train_p, train_observations)
    preds = test_df[["trip_id", "gps_point_id", "candidate_arc_id"]].copy()
    preds["true_label"] = test_df["label"].to_numpy().astype(int)
    preds["predicted_probability"] = p
    preds["top1_decision"] = top1_decision(test_df, p)
    preds["model_name"] = name
    return metrics, preds, fit_s, 1e6 * inf_s / max(len(test_df), 1), train_metrics
