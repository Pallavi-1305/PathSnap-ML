"""Optuna TPE search over training trips with trip-grouped cross-validation."""
from __future__ import annotations
import json
import time
from pathlib import Path

import optuna
import pandas as pd
import yaml

from .models import cross_validate


def suggest_params(trial: optuna.Trial, model: str) -> dict:
    if model == "logreg":
        return {"C": trial.suggest_float("C", 1e-3, 100.0, log=True)}
    if model == "naive_bayes":
        return {"var_smoothing": trial.suggest_float("var_smoothing", 1e-12, 1e-6, log=True)}
    if model == "decision_tree":
        return {"max_depth": trial.suggest_int("max_depth", 4, 32),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 100),
                "criterion": trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"])}
    if model == "random_forest":
        return {"n_estimators": trial.suggest_int("n_estimators", 50, 220, step=10),
                "max_depth": trial.suggest_int("max_depth", 5, 28),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 30),
                "max_samples": trial.suggest_float("max_samples", .2, 1.0),
                "max_features": trial.suggest_float("max_features", .35, 1.0),
                "criterion": trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"])}
    if model == "hist_gradient_boosting":
        return {"learning_rate": trial.suggest_float("learning_rate", .02, .3, log=True),
                "max_iter": trial.suggest_int("max_iter", 100, 350, step=25),
                "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 63),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 80),
                "l2_regularization": trial.suggest_float("l2_regularization", 1e-8, 10.0, log=True)}
    raise ValueError(f"No tuning space for model {model}")


def tune_model(model: str, candidates: pd.DataFrame, observations: pd.DataFrame,
               features: list[str], cfg: dict, output_dir: Path) -> dict:
    """Tune one already-selected classifier; this function never loads student data."""
    requested_folds = int(cfg.get("tuning", {}).get("cv_folds",
                         cfg.get("modelling", {}).get("cv_folds", 5)))
    if candidates.trip_id.nunique() < requested_folds:
        raise ValueError("There are fewer training trips than GroupKFold splits")
    if set(candidates.trip_id) & set(observations.loc[observations.partition != "train", "trip_id"]):
        raise ValueError("Non-training observations passed into Optuna")
    sampler = optuna.samplers.TPESampler(seed=int(cfg["seed"]))
    study = optuna.create_study(direction="maximize", sampler=sampler,
                                study_name=f"pathsnap-{model}-groupkfold")
    output_dir.mkdir(parents=True, exist_ok=True)
    trial_rows = []
    folds_cfg = dict(cfg["modelling"], seed=cfg["seed"], cv_folds=requested_folds)

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, model)
        folds = cross_validate(model, candidates, features, folds_cfg,
                               observations=observations, params=params)
        scores = folds.macro_f1.astype(float).tolist()
        trial.set_user_attr("fold_macro_f1", scores)
        trial.set_user_attr("mean_fit_time_s", float(folds.fit_time_s.mean()))
        return float(sum(scores) / len(scores))

    def record(study_: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        trial_rows.append({
            "trial_number": trial.number,
            "state": trial.state.name,
            "objective_value": trial.value,
            "duration_seconds": trial.duration.total_seconds() if trial.duration else None,
            "params_json": json.dumps(trial.params, sort_keys=True),
            "fold_macro_f1_json": json.dumps(trial.user_attrs.get("fold_macro_f1", [])),
            "mean_fit_time_s": trial.user_attrs.get("mean_fit_time_s"),
            "failure": trial.user_attrs.get("failure", ""),
        })
        pd.DataFrame(trial_rows).sort_values("trial_number").to_csv(output_dir / "tuning_trials.csv", index=False)

    def safe_objective(trial: optuna.Trial) -> float:
        started = time.perf_counter()
        try:
            return objective(trial)
        except Exception as exc:
            trial.set_user_attr("failure", f"{type(exc).__name__}: {exc}")
            raise
        finally:
            trial.set_user_attr("wall_seconds", time.perf_counter() - started)

    n_trials = int(cfg["tuning"].get("n_trials", 12))
    timeout = int(cfg["tuning"].get("timeout_seconds", 0)) or None
    study.optimize(safe_objective, n_trials=n_trials, timeout=timeout,
                   callbacks=[record], catch=(ValueError,))
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise RuntimeError("Optuna produced no completed trials; inspect results/tuning_trials.csv")
    best = study.best_trial
    result = {
        "selected_model": model,
        "objective": "mean training-only GroupKFold macro F1 (full observation miss accounting)",
        "cv_folds": folds_cfg["cv_folds"],
        "n_trials_requested": n_trials,
        "n_trials_completed": len(completed),
        "best_trial_number": best.number,
        "best_cv_macro_f1": float(best.value),
        "best_params": best.params,
        "fold_macro_f1": best.user_attrs.get("fold_macro_f1", []),
    }
    (output_dir / "best_params.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
