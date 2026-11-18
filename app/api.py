"""Laptop 1 inference/evaluation API. Reference-route data never leaves this process."""
from __future__ import annotations
import logging
import sys
import time
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from pathsnap.data import load_trip, load_route, preprocess_trip, build_road_network, calculate_distance
from pathsnap.hmm import DirectedRoadNetwork, TransitionModel, collapse_consecutive, route_metrics, viterbi_decode
from pathsnap.pipeline import account_candidates, resolve_raw_root, trip_folder, _nearest_route_arcs
from preprocessing.candidate_generation import (add_relative_features, attach_nearest_route_labels,
                                                  build_candidate_features)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s",
                    stream=sys.stdout)
logger = logging.getLogger("pathsnap.api")


def create_app(project_root: Path = ROOT) -> FastAPI:
    root = Path(project_root).resolve()
    app = FastAPI(title="PathSnap frozen map-matching API", version="1.0")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    @app.middleware("http")
    async def request_log(request, call_next):
        started = time.perf_counter()
        logger.info("request.start method=%s path=%s", request.method, request.url.path)
        try:
            response = await call_next(request)
            logger.info("request.complete method=%s path=%s status=%s elapsed_s=%.3f",
                        request.method, request.url.path, response.status_code,
                        time.perf_counter() - started)
            return response
        except Exception:
            logger.exception("request.failed method=%s path=%s elapsed_s=%.3f",
                             request.method, request.url.path, time.perf_counter() - started)
            raise

    def load_settings():
        cfg = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))
        meta_path = root / "artifacts/model_metadata.json"
        model_path = root / "artifacts/final_model.joblib"
        if not meta_path.exists() or not model_path.exists():
            raise HTTPException(status_code=503, detail="Frozen model artifact is missing; run scripts/4_export_final.py")
        import json
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        model = joblib.load(model_path)
        splits = pd.read_csv(root / cfg["data"]["splits_file"], dtype={"trip_id": str})
        allowed = set(splits.loc[splits.partition == "student_test", "trip_id"].astype(str))
        raw_root = resolve_raw_root(root, cfg["data"]["raw_root"])
        return cfg, metadata, model, allowed, raw_root

    @app.get("/health")
    def health():
        return {"status": "ok", "model_ready": (root / "artifacts/final_model.joblib").exists()}

    @app.get("/api/trips")
    def list_student_trips():
        try:
            cfg = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))
            splits = pd.read_csv(root / cfg["data"]["splits_file"], dtype={"trip_id": str})
            ids = sorted(splits.loc[splits.partition == "student_test", "trip_id"].astype(str))
            return {"trip_ids": ids}
        except Exception as exc:
            logger.exception("trip_list.failed")
            raise HTTPException(status_code=500, detail=f"Could not list student trips: {exc}") from exc

    @app.post("/api/decode/{trip_id}")
    def decode_trip(trip_id: str):
        logger.info("trip.request trip_id=%s", trip_id)
        started = time.perf_counter()
        try:
            cfg, metadata, model, allowed, raw_root = load_settings()
            if trip_id not in allowed:
                raise HTTPException(status_code=404, detail="Trip is not in the student-test allowlist")
            folder = trip_folder(raw_root, "student_test", trip_id)
            # Do not open the .route file until predictions have been generated.
            raw = load_trip(folder, trip_id=trip_id, include_route=False)
            trip = preprocess_trip(raw)
            crs = metadata.get("crs", cfg["data"]["crs"])
            network = build_road_network(trip["nodes"], trip["arcs"], crs=crs, build_graph=False)
            matching = cfg.get("matching", {})
            radius = float(metadata.get("candidate_radius_m") or matching.get("candidate_radius_m", 200))
            cap = int(metadata.get("candidate_cap") or matching.get("max_candidates", 30))
            candidates = build_candidate_features(
                trip_id=trip_id, track_df=trip["track"], nodes_df=trip["nodes"],
                arcs_df=trip["arcs"], search_radius_m=radius, max_candidates=cap,
                crs=crs, network=network)
            if not candidates.empty:
                candidates = add_relative_features(candidates)
            else:
                candidates = pd.DataFrame(columns=["trip_id", "gps_point_id", "candidate_arc_id", *metadata["feature_columns"]])
            features = metadata["feature_columns"]
            for col in features:
                if col not in candidates:
                    candidates[col] = np.nan
            probabilities = model.predict_proba(candidates[features].to_numpy(dtype=np.float32))[:, 1] if len(candidates) else np.array([])
            candidates = candidates.copy()
            candidates["predicted_probability"] = probabilities

            lengths = network["arcs"].set_index("arc_id").length_m.to_dict()
            state_network = DirectedRoadNetwork(network["arcs"].loc[network["arcs"].valid], lengths)
            hcfg = metadata.get("hmm", cfg["hmm"])
            tcfg = hcfg.get("transition", {})
            transition = TransitionModel(state_network,
                distance_scale_m=float(tcfg.get("distance_scale_m", 100)),
                max_network_distance_m=float(tcfg.get("max_network_distance_m", 5000)),
                max_speed_mps=float(tcfg.get("max_speed_mps", 70)))
            track = trip["track"].sort_values("gps_point_id").reset_index(drop=True)
            groups = {int(pid): frame for pid, frame in candidates.groupby("gps_point_id", sort=True)} if len(candidates) else {}
            observations, classifier_ids = [], []
            previous = None
            indexed = track.set_index("gps_point_id")
            for point in track.gps_point_id.astype(int):
                row = indexed.loc[point]
                group = groups.get(point)
                layer = [] if group is None else [
                    {"arc_id": int(r.candidate_arc_id), "probability": float(r.predicted_probability)}
                    for r in group.itertuples(index=False)]
                if previous is None:
                    displacement, elapsed = 0.0, None
                else:
                    prior = indexed.loc[previous]
                    displacement = calculate_distance((float(prior.lon), float(prior.lat)),
                                                      (float(row.lon), float(row.lat)))
                    gap = float(row.timestamp_s - prior.timestamp_s)
                    elapsed = gap if np.isfinite(gap) and gap > 0 else None
                for candidate in layer:
                    candidate["gps_displacement_m"] = displacement
                    candidate["elapsed_s"] = elapsed
                observations.append(layer)
                classifier_ids.append(max(layer, key=lambda c: c["probability"])["arc_id"] if layer else None)
                previous = int(point)
            decoded = viterbi_decode(observations, transition,
                probability_floor=float(hcfg.get("probability_floor", 1e-12)))

            # Evaluation-only reads start after both routes exist.
            route_file = next(folder.glob("*.route"), None)
            if route_file is None:
                raise HTTPException(status_code=422, detail="Student reference route is unavailable for evaluation")
            reference = load_route(route_file)
            candidates_eval = candidates.copy()
            candidates_eval = attach_nearest_route_labels(candidates_eval, reference) if len(candidates_eval) else candidates_eval.assign(label=pd.Series(dtype=int))
            ledger = track[["gps_point_id"]].copy()
            ledger.insert(0, "trip_id", trip_id)
            ref_arc, ref_dist = _nearest_route_arcs(track, network, reference)
            ledger["nearest_reference_arc_id"] = ref_arc
            ledger["nearest_reference_distance_m"] = ref_dist
            ledger = account_candidates(ledger, candidates_eval)
            missing = int(ledger.candidate_miss.sum())
            route_ref = reference.arc_id.astype(int).tolist()
            cls_route, hmm_route = collapse_consecutive(classifier_ids), collapse_consecutive(decoded)
            cls_metrics = route_metrics(cls_route, route_ref, missing_predictions=missing)
            hmm_metrics = route_metrics(hmm_route, route_ref, missing_predictions=missing)
            arcs_geo = network["arcs"].set_index("arc_id")
            nodes_geo = network["nodes"].set_index("node_id")
            def geometry(arc_ids):
                output = []
                for aid in arc_ids:
                    if aid is None or int(aid) not in arcs_geo.index:
                        continue
                    edge = arcs_geo.loc[int(aid)]
                    u, v = int(edge.from_node_id), int(edge.to_node_id)
                    if u in nodes_geo.index and v in nodes_geo.index:
                        a, b = nodes_geo.loc[u], nodes_geo.loc[v]
                        output.append({"arc_id": int(aid), "coordinates": [[float(a.lat), float(a.lon)],
                                                                               [float(b.lat), float(b.lon)]]})
                return output
            gps = [{"gps_point_id": int(r.gps_point_id), "lat": float(r.lat), "lon": float(r.lon),
                    "candidate_count": len(observations[i]),
                    "classifier_arc_id": classifier_ids[i], "hmm_arc_id": decoded[i]}
                   for i, r in enumerate(track.itertuples(index=False))]
            unique_candidates = sorted(set(candidates.candidate_arc_id.astype(int))) if len(candidates) else []
            response = {
                "trip_id": trip_id, "selected_model": metadata["model"],
                "candidate_radius_m": radius, "candidate_cap": cap,
                "observation_count": len(track), "missing_candidate_observations": missing,
                "candidate_miss_rate": missing / len(track) if len(track) else 0.0,
                "classifier_only_metrics": {**cls_metrics, "transition_feasibility": _continuity(cls_route, transition)},
                "hmm_metrics": {**hmm_metrics, "transition_feasibility": _continuity(hmm_route, transition)},
                "gps_points": gps, "candidate_segments": geometry(unique_candidates),
                "classifier_route_segments": geometry(cls_route), "hmm_route_segments": geometry(hmm_route),
                "reference_route_exposed": False,
            }
            logger.info("trip.complete trip_id=%s status=success observations=%d missing=%d elapsed_s=%.3f",
                        trip_id, len(track), missing, time.perf_counter() - started)
            return response
        except HTTPException as exc:
            logger.warning("trip.complete trip_id=%s status=%d elapsed_s=%.3f",
                           trip_id, exc.status_code, time.perf_counter() - started)
            raise
        except Exception as exc:
            logger.exception("trip.complete trip_id=%s status=500 elapsed_s=%.3f",
                             trip_id, time.perf_counter() - started)
            raise HTTPException(status_code=500, detail=f"Trip processing failed: {type(exc).__name__}") from exc

    return app


def _continuity(arcs: list[int], model: TransitionModel) -> float:
    if len(arcs) < 2:
        return 1.0
    return sum(model.score(a, b).reachable for a, b in zip(arcs, arcs[1:])) / (len(arcs) - 1)


app = create_app()


def main() -> None:
    import uvicorn

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    api_cfg = cfg.get("api", {})
    host = str(api_cfg.get("host", "127.0.0.1"))
    port = int(api_cfg.get("port", 8000))
    certfile = str(api_cfg.get("ssl_certfile", "")).strip()
    keyfile = str(api_cfg.get("ssl_keyfile", "")).strip()

    loopback_hosts = {"127.0.0.1", "::1", "localhost"}
    if host not in loopback_hosts and (not certfile or not keyfile):
        raise RuntimeError(
            "HTTPS certificates are required when api.host exposes the server beyond loopback."
        )

    tls = {}
    if certfile or keyfile:
        if not certfile or not keyfile:
            raise RuntimeError("api.ssl_certfile and api.ssl_keyfile must both be configured.")
        cert_path = (ROOT / certfile).resolve()
        key_path = (ROOT / keyfile).resolve()
        if not cert_path.is_file():
            raise RuntimeError(f"TLS certificate not found: {cert_path}")
        if not key_path.is_file():
            raise RuntimeError(f"TLS private key not found: {key_path}")
        tls = {"ssl_certfile": str(cert_path), "ssl_keyfile": str(key_path)}

    uvicorn.run(app, host=host, port=port, **tls)


if __name__ == "__main__":
    main()
