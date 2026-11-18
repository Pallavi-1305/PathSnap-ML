"""Run Optuna tuning on the CV-selected model using training trips only."""
from __future__ import annotations
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pathsnap.models import check_columns
from pathsnap.tune import tune_model


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    selected_path = ROOT / "results/selected_model.json"
    if not selected_path.exists():
        raise FileNotFoundError("Run scripts/2_compare_models.py before tuning")
    model = json.loads(selected_path.read_text(encoding="utf-8"))["selected_model"]
    path = ROOT / cfg["data"]["candidates_file"]
    obs_path = path.parent / "observation_accounting.csv"
    if not path.exists() or not obs_path.exists():
        raise FileNotFoundError("Run run_preprocessing.py before tuning")
    splits = pd.read_csv(ROOT / cfg["data"]["splits_file"], dtype={"trip_id": str})
    train_ids = set(splits.loc[splits.partition == "train", "trip_id"].astype(str))
    candidates = pd.read_csv(path, dtype={"trip_id": str})
    observations = pd.read_csv(obs_path, dtype={"trip_id": str})
    candidates = candidates[candidates.trip_id.isin(train_ids)].reset_index(drop=True)
    observations = observations[observations.trip_id.isin(train_ids)].reset_index(drop=True)
    if set(candidates.trip_id) != train_ids:
        missing = sorted(train_ids - set(candidates.trip_id))
        raise ValueError(f"Some train trips have no candidates at selected radius: {missing[:5]}")
    check_columns(candidates, cfg["features"], "Optuna training data")
    result = tune_model(model, candidates, observations, cfg["features"], cfg, ROOT / "results")
    print(f"Best training-only Optuna trial {result['best_trial_number']}: "
          f"CV macro F1={result['best_cv_macro_f1']:.4f}")
    print(f"Parameters: {result['best_params']}")
    print("Student-test data was not loaded or scored by the tuner.")


if __name__ == "__main__":
    main()
