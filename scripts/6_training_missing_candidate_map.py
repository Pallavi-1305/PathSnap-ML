"""Create a diagnostic Folium example when student trips have no empty layers.

This map uses only training trip 00000002 and a frozen prediction model; it is
not an evaluation result and does not affect configuration selection.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pathsnap.data import load_trip, load_route, preprocess_trip, build_road_network, calculate_distance
from pathsnap.hmm import DirectedRoadNetwork, TransitionModel, viterbi_decode
from pathsnap.maps import save_folium_route_map
from pathsnap.pipeline import resolve_raw_root, trip_folder
from preprocessing.candidate_generation import add_relative_features, build_candidate_features


def main():
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    metadata = json.loads((ROOT / "artifacts/model_metadata.json").read_text(encoding="utf-8"))
    model = joblib.load(ROOT / "artifacts/final_model.joblib")
    raw_root = resolve_raw_root(ROOT, cfg["data"]["raw_root"])
    trip_id = "00000002"
    folder = trip_folder(raw_root, "train", trip_id)
    # The reference is omitted until classifier and Viterbi predictions exist.
    raw = load_trip(folder, trip_id=trip_id, include_route=False)
    trip = preprocess_trip(raw)
    network = build_road_network(trip["nodes"], trip["arcs"], crs=metadata["crs"], build_graph=False)
    candidates = build_candidate_features(
        trip_id=trip_id, track_df=trip["track"], nodes_df=trip["nodes"], arcs_df=trip["arcs"],
        search_radius_m=float(metadata["candidate_radius_m"]),
        max_candidates=int(metadata["candidate_cap"]), crs=metadata["crs"], network=network)
    if not candidates.empty:
        candidates = add_relative_features(candidates)
        for feature in metadata["feature_columns"]:
            if feature not in candidates:
                candidates[feature] = np.nan
        candidates["probability"] = model.predict_proba(
            candidates[metadata["feature_columns"]].to_numpy(dtype=np.float32))[:, 1]
    else:
        candidates["probability"] = pd.Series(dtype=float)
    track = trip["track"].sort_values("gps_point_id").reset_index(drop=True)
    grouped = {int(pid): frame for pid, frame in candidates.groupby("gps_point_id")} if len(candidates) else {}
    layers, classifier = [], []
    previous = None
    indexed = track.set_index("gps_point_id")
    for pid in track.gps_point_id.astype(int):
        row = indexed.loc[pid]
        group = grouped.get(pid)
        layer = [] if group is None else [
            {"arc_id": int(r.candidate_arc_id), "probability": float(r.probability)}
            for r in group.itertuples(index=False)]
        if previous is None:
            displacement, elapsed = 0.0, None
        else:
            prior = indexed.loc[previous]
            displacement = calculate_distance((float(prior.lon), float(prior.lat)),
                                              (float(row.lon), float(row.lat)))
            gap = float(row.timestamp_s - prior.timestamp_s)
            elapsed = gap if np.isfinite(gap) and gap > 0 else None
        for state in layer:
            state["gps_displacement_m"], state["elapsed_s"] = displacement, elapsed
        layers.append(layer)
        classifier.append(max(layer, key=lambda state: state["probability"])["arc_id"] if layer else None)
        previous = int(pid)
    lengths = network["arcs"].set_index("arc_id").length_m.to_dict()
    directed = DirectedRoadNetwork(network["arcs"].loc[network["arcs"].valid], lengths)
    hcfg = metadata["hmm"]; tcfg = hcfg.get("transition", {})
    transition = TransitionModel(directed,
        distance_scale_m=float(tcfg.get("distance_scale_m", 100)),
        max_network_distance_m=float(tcfg.get("max_network_distance_m", 5000)),
        max_speed_mps=float(tcfg.get("max_speed_mps", 70)))
    decoded = viterbi_decode(layers, transition,
        probability_floor=float(hcfg.get("probability_floor", 1e-12)))
    # Training reference is read only after predictions, solely for the map layer.
    reference = load_route(next(folder.glob("*.route")))
    missing = {int(pid) for pid, layer in zip(track.gps_point_id, layers) if not layer}
    out = ROOT / "results/figures/maps/missing_candidates_training_00000002.html"
    save_folium_route_map(track, network, candidates,
        [int(a) for a in classifier if a is not None],
        [int(a) for a in decoded if a is not None], reference.arc_id.astype(int).tolist(),
        out, title=f"Training diagnostic {trip_id}; no candidate at {len(missing)} GPS points",
        missing_gps_ids=missing)
    print(f"Saved {out}; {len(missing)} of {len(track)} observations have no candidates.")


if __name__ == "__main__":
    main()
