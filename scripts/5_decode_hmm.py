"""Select best cross-validation model and evaluate classifier vs HMM on student_test."""
from __future__ import annotations
import argparse
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pathsnap.data import load_trip, load_route, preprocess_trip, build_road_network, calculate_distance
from pathsnap.hmm import DirectedRoadNetwork, TransitionModel, collapse_consecutive, route_metrics, viterbi_decode
from preprocessing.road_network import project_coords
from pathsnap.maps import save_folium_route_map


def select_model(cv: pd.DataFrame, requested: str | None) -> tuple[str, pd.DataFrame]:
    if not {"model", "macro_f1"}.issubset(cv.columns):
        raise ValueError("CV results must contain model and macro_f1")
    means = cv.groupby("model", as_index=False)["macro_f1"].mean()
    if means.empty or means["macro_f1"].isna().all():
        raise ValueError("No valid training-only CV macro F1 scores")
    best = str(means.sort_values(["macro_f1", "model"], ascending=[False, True]).iloc[0]["model"])
    selected = requested or best
    if selected not in set(means["model"]):
        raise ValueError(f"Model {selected!r} is absent from CV results")
    return selected, means


def find_trip_folder(raw_root: Path, partition: str, trip_id: str) -> Path:
    direct = raw_root / partition / str(trip_id)
    if direct.is_dir():
        return direct
    # Also allow a raw root that is itself the partition folder.
    direct = raw_root / str(trip_id)
    if direct.is_dir():
        return direct
    raise FileNotFoundError(f"Trip folder missing for {partition} trip {trip_id}: searched {raw_root}")


def continuity(arcs: list[int], model: TransitionModel) -> float:
    if len(arcs) < 2:
        return 1.0
    valid = sum(model.score(a, b).reachable for a, b in zip(arcs, arcs[1:]))
    return valid / (len(arcs) - 1)


def make_route_figure(trip_id: str, track: pd.DataFrame, network: dict,
                      candidates: pd.DataFrame, classifier: list[int],
                      decoded: list[int], reference: list[int], out: Path) -> None:
    """Draw an example in the project's projected CRS; no evaluator data are read."""
    arcs = network["arcs"].set_index("arc_id")
    fig, ax = plt.subplots(figsize=(10, 8))
    cand_ids = sorted(set(candidates["candidate_arc_id"].astype(int)))
    for aid in cand_ids:
        if aid in arcs.index:
            row = arcs.loc[aid]
            ax.plot([row.from_x, row.to_x], [row.from_y, row.to_y],
                    color="#c8c8c8", linewidth=0.35, alpha=0.25)
    def draw(ids, color, label, width):
        first = True
        for aid in ids:
            if aid not in arcs.index:
                continue
            row = arcs.loc[aid]
            ax.plot([row.from_x, row.to_x], [row.from_y, row.to_y],
                    color=color, linewidth=width, label=label if first else None,
                    zorder=3)
            first = False
    draw(reference, "#2ca02c", "Reference route", 2.6)
    draw(classifier, "#ff7f0e", "Classifier only", 1.5)
    draw(decoded, "#1f77b4", "Classifier + HMM", 2.1)
    gps = track.dropna(subset=["x", "y"])
    ax.scatter(gps["x"], gps["y"], s=9, c="black", label="GPS observations", zorder=5)
    ax.set_title(f"Example student-test trip {trip_id}")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.axis("equal")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Override model; default is highest mean training GroupKFold macro F1")
    parser.add_argument("--raw-root", type=Path, help="Root containing train/ and student_test/ folders")
    parser.add_argument("--predictions", type=Path, default=ROOT / "results/candidate_probabilities.csv")
    parser.add_argument("--observation-file", type=Path, default=ROOT / "data/processed/observation_accounting.csv")
    parser.add_argument("--cv-results", type=Path, default=ROOT / "results/cv_fold_results.csv")
    parser.add_argument("--splits", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    dcfg = cfg["data"]
    raw_root = args.raw_root or (ROOT / dcfg.get("raw_root", "../PathSnap-Dataset/Student folder")).resolve()
    splits_path = args.splits or ROOT / dcfg["splits_file"]
    cv_path, pred_path, observation_path = args.cv_results, args.predictions, args.observation_file
    for path in (splits_path, cv_path, pred_path, observation_path):
        if not path.exists():
            raise FileNotFoundError(f"Required input missing: {path}")
    splits = pd.read_csv(splits_path, dtype={"trip_id": str})
    if not {"trip_id", "partition"}.issubset(splits.columns):
        raise ValueError("splits file needs trip_id, partition columns")
    if splits["trip_id"].duplicated().any():
        raise ValueError("splits file contains duplicate trip IDs")
    cv, pred = pd.read_csv(cv_path), pd.read_csv(pred_path, dtype={"trip_id": str})
    obs_all = pd.read_csv(observation_path, dtype={"trip_id": str})
    hcfg = cfg.get("hmm", {})
    configured_model = hcfg.get("selected_classifier", "auto")
    requested_model = args.model or (None if configured_model == "auto" else configured_model)
    selected, cv_means = select_model(cv, requested_model)
    model_column = "selected_model" if "selected_model" in pred.columns else "model_name"
    needed = {"trip_id", "gps_point_id", "candidate_arc_id", "predicted_probability", model_column}
    if not needed.issubset(pred.columns):
        raise ValueError(f"prediction file is missing {sorted(needed - set(pred.columns))}")
    if "student_test" not in set(splits["partition"]):
        raise ValueError("splits file has no student_test partition")
    student_ids = set(splits.loc[splits.partition == "student_test", "trip_id"].astype(str))
    unknown = set(pred.trip_id.astype(str)) - set(splits.trip_id.astype(str))
    if unknown:
        raise ValueError(f"Prediction file includes IDs absent from splits.csv: {sorted(unknown)[:5]}")
    if selected not in set(pred[model_column].astype(str)):
        # An override can request a non-default model; use the all-model score
        # table, while selecting only its probabilities and never its labels.
        all_model_path = ROOT / "results/student_test_predictions.csv"
        if not all_model_path.exists():
            raise ValueError(f"Selected model {selected} is not in the probability export")
        pred = pd.read_csv(all_model_path, usecols=["trip_id", "gps_point_id", "candidate_arc_id",
                                                   "predicted_probability", "model_name"],
                           dtype={"trip_id": str})
        model_column = "model_name"
    pred = pred[(pred[model_column].astype(str) == selected) & pred.trip_id.astype(str).isin(student_ids)].copy()
    if pred.empty:
        raise ValueError(f"No student-test predictions found for selected model {selected}")
    pred["trip_id"] = pred["trip_id"].astype(str)
    pred["gps_point_id"] = pred["gps_point_id"].astype(int)
    pred["candidate_arc_id"] = pred["candidate_arc_id"].astype(int)
    pred["predicted_probability"] = pd.to_numeric(pred.predicted_probability, errors="coerce")
    transition_cfg = hcfg.get("transition", {})
    output_dir = args.output_dir
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    prediction_rows, evaluation_rows = [], []
    map_registry = {}
    for tid in sorted(student_ids):
        sub = pred[pred.trip_id == tid].sort_values(["gps_point_id", "candidate_arc_id"])
        if sub.empty:
            raise ValueError(f"No predictions for student-test trip {tid} using {selected}")
        folder = find_trip_folder(raw_root, "student_test", tid)
        # Prediction-time trip load intentionally excludes the .route file.
        trip = load_trip(folder, trip_id=tid, require_route=False, include_route=False)
        prepared = preprocess_trip(trip)
        track = prepared["track"].copy()
        road = build_road_network(trip["nodes"], trip["arcs"], crs=cfg.get("data", {}).get("crs", "EPSG:27700"), build_graph=False)
        track["x"], track["y"] = project_coords(track["lon"].to_numpy(), track["lat"].to_numpy(),
                                                  target_crs=road["crs"], transformer=road.get("transformer"))
        valid_arc = road["arcs"].set_index("arc_id")["length_m"].to_dict()
        net = DirectedRoadNetwork(road["arcs"].loc[road["arcs"]["valid"]], valid_arc)
        trans = TransitionModel(net,
            distance_scale_m=float(transition_cfg.get("distance_scale_m", 100.0)),
            max_network_distance_m=float(transition_cfg.get("max_network_distance_m", 5000.0)),
            max_speed_mps=float(transition_cfg.get("max_speed_mps", 70.0)))
        grouped = {int(pid): g for pid, g in sub.groupby("gps_point_id", sort=True)}
        point_info = track.set_index("gps_point_id")
        observations, classifier_ids = [], []
        ordered_ids = sorted(map(int, track.gps_point_id))
        previous = None
        for pid in ordered_ids:
            gps = point_info.loc[pid]
            candidates = grouped.get(pid)
            layer = []
            if candidates is not None:
                for row in candidates.itertuples(index=False):
                    aid = int(row.candidate_arc_id)
                    layer.append({"arc_id": aid, "probability": row.predicted_probability})
            if previous is None:
                displacement, elapsed = 0.0, None
            else:
                prev = point_info.loc[previous]
                displacement = calculate_distance((float(prev.lon), float(prev.lat)),
                                                   (float(gps.lon), float(gps.lat)))
                dt = float(gps.timestamp_s - prev.timestamp_s)
                elapsed = dt if math.isfinite(dt) and dt > 0 else None
            for item in layer:
                item["gps_displacement_m"] = displacement
                item["elapsed_s"] = elapsed
            observations.append(layer)
            classifier_ids.append(max(layer, key=lambda c: (-math.inf if pd.isna(c["probability"]) else c["probability"]))["arc_id"] if layer else None)
            previous = pid
        decoded = viterbi_decode(observations, trans,
             probability_floor=float(hcfg.get("probability_floor", 1e-12)))
        classifier_route = collapse_consecutive(classifier_ids)
        hmm_route = collapse_consecutive(decoded)
        # Reference route is opened only after both predictions have been generated.
        route_file = next(folder.glob("*.route"), None)
        if route_file is None:
            raise FileNotFoundError(f"Reference route missing for evaluation trip {tid}")
        reference = load_route(route_file)["arc_id"].astype(int).tolist()
        trip_obs = obs_all[(obs_all.trip_id.astype(str) == tid) &
                           (obs_all.partition == "student_test")].sort_values("gps_point_id")
        if len(trip_obs) != len(ordered_ids):
            raise ValueError(f"Observation accounting mismatch for trip {tid}: {len(trip_obs)} vs {len(ordered_ids)}")
        misses = int(trip_obs.candidate_miss.astype(bool).sum())
        valid_truth = trip_obs.nearest_reference_arc_id.notna() & (trip_obs.nearest_reference_arc_id >= 0)
        for approach, route in (("classifier_only", classifier_route), ("classifier_hmm_viterbi", hmm_route)):
            arc_predictions = classifier_ids if approach == "classifier_only" else decoded
            route_hit_count = sum(
                valid_truth.iloc[pos] and pred_arc is not None and int(pred_arc) == int(true_arc)
                for pos, (pred_arc, true_arc) in enumerate(zip(arc_predictions, trip_obs.nearest_reference_arc_id)))
            point_arc_accuracy = route_hit_count / len(trip_obs) if len(trip_obs) else 0.0
            row = {"trip_id": tid, "approach": approach,
                   **route_metrics(route, reference, missing_predictions=misses),
                   "transition_feasibility": continuity(route, trans),
                   "candidate_miss_rate": misses / len(trip_obs) if len(trip_obs) else 0.0,
                   "observation_arc_accuracy": point_arc_accuracy,
                   "original_observation_count": len(trip_obs),
                   "missing_candidate_observations": misses,
                   "zero_candidate_observations": sum(not layer for layer in observations),
                   "predicted_arc_count": len(route), "reference_arc_count": len(reference)}
            evaluation_rows.append(row)
        for obs_index, (pid, candidates, c_arc, h_arc) in enumerate(zip(ordered_ids, observations, classifier_ids, decoded)):
            probs = [x["probability"] for x in candidates if pd.notna(x.get("probability"))]
            prob_by_arc = {x["arc_id"]: x["probability"] for x in candidates}
            point = point_info.loc[pid]
            prediction_rows.append({
                "trip_id": tid, "gps_point_id": pid, "timestamp_s": point.timestamp_s,
                "classifier_only_predicted_arc": c_arc, "hmm_decoded_predicted_arc": h_arc,
                "selected_classifier": selected,
                "classifier_probability": prob_by_arc.get(c_arc),
                "hmm_state_probability": prob_by_arc.get(h_arc),
                "candidate_count": len(candidates),
                "candidate_gap": len(candidates) == 0,
            })
        map_registry[tid] = {"folder": folder, "candidates": sub,
                             "classifier": classifier_route, "hmm": hmm_route,
                             "reference": reference, "missing": misses,
                             "zero_candidates": sum(not layer for layer in observations)}
    ev = pd.DataFrame(evaluation_rows)
    pred_out = pd.DataFrame(prediction_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_out.to_csv(output_dir / "hmm_predictions.csv", index=False)
    ev.to_csv(output_dir / "decoding_comparison.csv", index=False)
    metric_names = ["arc_precision", "arc_recall", "arc_f1", "normalized_sequence_similarity"]
    means = ev.groupby("approach")[metric_names].mean().reindex(["classifier_only", "classifier_hmm_viterbi"])
    fig, ax = plt.subplots(figsize=(9, 5))
    means.T.plot(kind="bar", ax=ax, color=["#e08e45", "#3b6ea5"])
    ax.set_xticklabels(["Precision", "Recall", "F1", "Sequence similarity"], rotation=0)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean per-trip score")
    ax.set_title(f"Student-test route metrics (selected model: {selected})")
    ax.legend(["Classifier only", "Classifier + HMM + Viterbi"])
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "route_metrics_comparison.png", dpi=160)
    plt.close(fig)
    cont = ev.groupby("approach")["transition_feasibility"].mean().reindex(["classifier_only", "classifier_hmm_viterbi"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(["Classifier only", "Classifier + HMM + Viterbi"], cont.values, color=["#e08e45", "#3b6ea5"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share of consecutive route transitions reachable")
    ax.set_title("Directed road-network continuity (student-test trips)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "route_continuity_comparison.png", dpi=160)
    plt.close(fig)
    # Static overview is retained, and the three requested interactive maps use
    # student-test routes only after prediction/evaluation has completed.
    best_delta = ev.pivot(index="trip_id", columns="approach", values="arc_f1")
    deltas = best_delta["classifier_hmm_viterbi"] - best_delta["classifier_only"]
    success_id = str(deltas.idxmax())
    no_improve = deltas[deltas <= 0]
    no_improve_id = str(no_improve.idxmin()) if len(no_improve) else str(deltas.idxmin())
    selected_maps = {"hmm_improvement": success_id, "hmm_no_improvement": no_improve_id}
    no_candidate_trips = [key for key, value in map_registry.items() if value["zero_candidates"] > 0]
    if no_candidate_trips:
        missing_id = str(max(no_candidate_trips,
                             key=lambda key: map_registry[key]["zero_candidates"]))
        selected_maps["missing_candidates"] = missing_id
    maps_dir = figures / "maps"; maps_dir.mkdir(parents=True, exist_ok=True)
    for map_kind, tid in selected_maps.items():
        item = map_registry[tid]
        map_trip = load_trip(item["folder"], trip_id=tid, include_route=False)
        map_prepared = preprocess_trip(map_trip)
        map_road = build_road_network(map_trip["nodes"], map_trip["arcs"],
                                     crs=cfg.get("data", {}).get("crs", "EPSG:32640"), build_graph=False)
        map_obs = pred_out[pred_out.trip_id.astype(str) == tid].sort_values("gps_point_id")
        save_folium_route_map(map_prepared["track"], map_road, item["candidates"],
                              item["classifier"], item["hmm"], item["reference"],
                              maps_dir / f"{map_kind}_{tid}.html", title=f"{map_kind}: trip {tid}",
                              missing_gps_ids=set(map_obs.loc[map_obs.candidate_gap, "gps_point_id"].astype(int)))
        if map_kind == "hmm_improvement":
            make_route_figure(tid, map_prepared["track"].assign(**{
                "x": project_coords(map_prepared["track"].lon.to_numpy(), map_prepared["track"].lat.to_numpy(),
                                    target_crs=map_road["crs"], transformer=map_road.get("transformer"))[0],
                "y": project_coords(map_prepared["track"].lon.to_numpy(), map_prepared["track"].lat.to_numpy(),
                                    target_crs=map_road["crs"], transformer=map_road.get("transformer"))[1]}),
                              map_road, item["candidates"], item["classifier"], item["hmm"],
                              item["reference"], figures / "example_route.png")
    print(f"Selected classifier: {selected} (training GroupKFold macro F1={float(cv_means.loc[cv_means.model == selected, 'macro_f1'].iloc[0]):.4f})")
    print(ev.groupby("approach")[metric_names + ["transition_feasibility", "candidate_miss_rate"]].mean().to_string())
    print(f"Saved: {output_dir / 'decoding_comparison.csv'}")
    print(f"Saved: {output_dir / 'hmm_predictions.csv'}")
    print(f"Saved figures under: {figures}")
    for kind, tid in selected_maps.items():
        print(f"Interactive {kind} map: {maps_dir / f'{kind}_{tid}.html'}")
    if not no_candidate_trips:
        print("No student-test trip has a zero-candidate observation at the frozen radius; "
              "scripts/6_training_missing_candidate_map.py creates the training-only diagnostic map.")

if __name__ == "__main__":
    main()
