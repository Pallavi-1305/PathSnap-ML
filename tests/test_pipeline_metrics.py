"""Full-observation accounting, strict labels, and group-split regression tests."""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from pathsnap.models import candidate_metrics, evaluate
from pathsnap.pipeline import account_candidates, filter_candidate_radius
from pathsnap.probabilities import candidate_probability_export
from pathsnap.tune import tune_model


def test_zero_candidate_observations_remain_explicit_misses():
    observations = pd.DataFrame({"trip_id": ["a"] * 3, "partition": ["train"] * 3,
                                 "gps_point_id": [0, 1, 2], "nearest_reference_arc_id": [4, 5, 6]})
    candidates = pd.DataFrame({"trip_id": ["a", "a"], "gps_point_id": [0, 1], "candidate_arc_id": [4, 8],
                               "label": [1, 0]})
    result = account_candidates(observations, candidates)
    assert result.candidate_count.tolist() == [1, 1, 0]
    assert result.has_candidate.tolist() == [True, True, False]
    assert result.has_correct_candidate.tolist() == [True, False, False]
    assert result.candidate_miss.tolist() == [False, True, True]


def test_accounting_keys_candidates_by_trip_and_observation():
    obs = pd.DataFrame({"trip_id": ["a", "b"], "gps_point_id": [1, 1]})
    cand = pd.DataFrame({"trip_id": ["a"], "gps_point_id": [1], "label": [1]})
    accounted = account_candidates(obs, cand)
    assert accounted.candidate_count.tolist() == [1, 0]
    assert accounted.has_correct_candidate.tolist() == [True, False]


def test_radius_filter_applies_radius_and_cap_before_strict_relabeling():
    candidates = pd.DataFrame({
        "trip_id": ["x"] * 4, "gps_point_id": [0, 0, 0, 1],
        "candidate_arc_id": [3, 4, 5, 7], "distance_m": [10., 20., 180., 40.],
    })
    route = pd.DataFrame({"arc_id": [4, 7]})
    selected = filter_candidate_radius(candidates, 150, 1, route)
    assert selected.candidate_arc_id.tolist() == [3, 7]
    assert selected.label.tolist() == [0, 1]
    assert selected.groupby("gps_point_id").size().max() == 1


def test_candidate_radius_configurations_filter_the_same_superset():
    candidates = pd.DataFrame({"trip_id": ["x"] * 3, "gps_point_id": [1, 1, 1],
        "candidate_arc_id": [4, 5, 6], "distance_m": [40., 120., 190.]})
    route = pd.DataFrame({"arc_id": [5, 6]})
    sizes = {radius: len(filter_candidate_radius(candidates, radius, 30, route))
             for radius in (50, 150, 200)}
    assert sizes == {50: 1, 150: 2, 200: 3}


def test_probability_export_uses_full_identifiers_and_excludes_labels():
    candidates = pd.DataFrame({"trip_id": ["a", "b"], "gps_point_id": [3, 3],
        "candidate_arc_id": [9, 9], "label": [1, 0]})
    exported = candidate_probability_export(candidates, [.7, .2], "logreg")
    assert exported[["trip_id", "gps_point_id", "candidate_arc_id"]].values.tolist() == [
        ["a", 3, 9], ["b", 3, 9]]
    assert exported.predicted_probability.tolist() == [.7, .2]
    assert set(exported.selected_model) == {"logreg"}
    assert "label" not in exported


def test_missed_observation_is_added_as_false_negative_to_metrics():
    result = candidate_metrics([1, 0], [.8, .2], [1, 1], missed_observations=1)
    assert result["confusion_fn"] == 1
    assert result["confusion_fp"] == 1
    assert abs(result["accuracy"] - 1 / 3) < 1e-12
    assert result["recall"] == 0.5
    assert result["macro_f1"] < 1.0


def test_observation_accuracy_denominator_includes_missing_candidate_point():
    df = pd.DataFrame({"trip_id": ["a", "a"], "gps_point_id": [0, 1],
                       "candidate_arc_id": [3, 8], "label": [1, 0]})
    obs = pd.DataFrame({"trip_id": ["a", "a", "a"], "gps_point_id": [0, 1, 2],
                        "has_correct_candidate": [True, False, False],
                        "candidate_count": [1, 1, 0],
                        "nearest_reference_arc_id": [3, 7, 6]})
    result = evaluate(df, np.array([.9, .7]), obs)
    assert result["observation_count"] == 3
    assert result["candidate_miss_rate"] == 2 / 3
    assert result["top1_acc"] == 1 / 3
    assert result["observation_arc_accuracy"] == 1 / 3


def test_group_kfold_keeps_each_trip_in_exactly_one_validation_fold():
    rows = pd.DataFrame({"trip_id": np.repeat([f"t{i}" for i in range(10)], 3)})
    seen = []
    for train_idx, val_idx in GroupKFold(n_splits=5).split(rows, groups=rows.trip_id):
        train_ids = set(rows.iloc[train_idx].trip_id)
        val_ids = set(rows.iloc[val_idx].trip_id)
        assert not train_ids.intersection(val_ids)
        seen.extend(val_ids)
    assert len(seen) == rows.trip_id.nunique()
    assert len(set(seen)) == len(seen)


def test_optuna_records_trial_parameters_objective_and_state(tmp_path):
    rows = []
    observations = []
    for i in range(6):
        tid = f"t{i}"
        rows.extend([
            {"trip_id": tid, "gps_point_id": 0, "candidate_arc_id": 10, "distance_m": 0.0, "label": 1},
            {"trip_id": tid, "gps_point_id": 0, "candidate_arc_id": 11, "distance_m": 10.0, "label": 0},
        ])
        observations.append({"trip_id": tid, "partition": "train", "gps_point_id": 0,
                             "has_correct_candidate": True, "candidate_count": 2})
    cfg = {"seed": 2, "modelling": {"cv_folds": 2, "neg_per_obs": 1},
           "tuning": {"cv_folds": 2, "n_trials": 1, "timeout_seconds": 0}}
    best = tune_model("random_forest", pd.DataFrame(rows), pd.DataFrame(observations),
                      ["distance_m"], cfg, tmp_path)
    recorded = pd.read_csv(tmp_path / "tuning_trials.csv")
    assert len(recorded) == 1
    assert recorded.loc[0, "state"] == "COMPLETE"
    assert recorded.loc[0, "objective_value"] == best["best_cv_macro_f1"]
    assert "n_estimators" in best["best_params"]
