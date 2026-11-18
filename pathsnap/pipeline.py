"""Shared trip loading, candidate generation, and full-observation accounting."""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from preprocessing.data_loader import load_trip
from preprocessing.preprocess import preprocess_trip
from preprocessing.road_network import build_road_network, project_coords
from preprocessing.candidate_generation import (
    add_relative_features, attach_nearest_route_labels, build_candidate_features,
)


def resolve_raw_root(project_root: Path, configured: str | Path) -> Path:
    path = Path(configured)
    return (project_root / path).resolve() if not path.is_absolute() else path


def trip_folder(raw_root: Path, partition: str, trip_id: str) -> Path:
    folder = raw_root / partition / str(trip_id)
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing {partition} trip {trip_id}: {folder}")
    return folder


def _nearest_route_arcs(track: pd.DataFrame, network: dict,
                        route: pd.DataFrame | None) -> tuple[np.ndarray, np.ndarray]:
    """Exact nearest route-segment distance using a midpoint KD-tree bound."""
    n = len(track)
    ids = np.full(n, -1, dtype=np.int64)
    distances = np.full(n, np.nan, dtype=float)
    if route is None or route.empty:
        return ids, distances
    route_ids = np.asarray(sorted(set(route.arc_id.astype(int))), dtype=np.int64)
    arcs = network["arcs"]
    roads = arcs[arcs.arc_id.isin(route_ids) & arcs.valid & arcs.length_m.notna()]
    if roads.empty:
        return ids, distances
    ax = roads.from_x.to_numpy(float); ay = roads.from_y.to_numpy(float)
    bx = roads.to_x.to_numpy(float); by = roads.to_y.to_numpy(float)
    arc_ids = roads.arc_id.to_numpy(np.int64)
    mid = np.column_stack(((ax + bx) / 2, (ay + by) / 2))
    half = float(np.max(roads.length_m.to_numpy(float))) / 2
    tree = cKDTree(mid)
    gps_x, gps_y = project_coords(track.lon.to_numpy(float), track.lat.to_numpy(float),
                                  target_crs=network["crs"], transformer=network.get("transformer"))
    for row_i, (x, y) in enumerate(zip(gps_x, gps_y)):
        if not bool(track.iloc[row_i].coord_valid):
            continue
        _, nearest = tree.query([x, y], k=1)
        ix = int(nearest)
        dx, dy = bx[ix] - ax[ix], by[ix] - ay[ix]
        den = dx * dx + dy * dy
        t = np.clip(((x - ax[ix]) * dx + (y - ay[ix]) * dy) / den, 0, 1) if den else 0.0
        upper = float(np.hypot(x - (ax[ix] + t * dx), y - (ay[ix] + t * dy)))
        nearby = np.asarray(tree.query_ball_point([x, y], r=upper + half), dtype=int)
        dxs, dys = bx[nearby] - ax[nearby], by[nearby] - ay[nearby]
        lensq = dxs * dxs + dys * dys
        ts = np.clip(((x - ax[nearby]) * dxs + (y - ay[nearby]) * dys) /
                     np.where(lensq == 0, 1.0, lensq), 0, 1)
        ds = np.hypot(x - (ax[nearby] + ts * dxs), y - (ay[nearby] + ts * dys))
        best_i = int(nearby[int(np.argmin(ds))])
        ids[row_i] = arc_ids[best_i]
        distances[row_i] = float(np.min(ds))
    return ids, distances


def account_candidates(observations: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    """Add per-original-observation candidate counts and strict-label misses."""
    obs = observations.copy()
    if candidates.empty:
        counts = pd.Series(dtype=np.int64, index=pd.MultiIndex.from_arrays([[], []], names=["trip_id", "gps_point_id"]))
        correct = pd.Series(dtype=bool, index=pd.MultiIndex.from_arrays([[], []], names=["trip_id", "gps_point_id"]))
    else:
        keys = ["trip_id", "gps_point_id"]
        counts = candidates.groupby(keys).size()
        correct = candidates.groupby(keys)["label"].max().eq(1)
    keys = pd.MultiIndex.from_frame(obs[["trip_id", "gps_point_id"]].assign(
        trip_id=obs["trip_id"].astype(str), gps_point_id=obs["gps_point_id"].astype(int)))
    obs["candidate_count"] = counts.reindex(keys).fillna(0).to_numpy(dtype=int)
    obs["has_candidate"] = obs.candidate_count.gt(0)
    obs["has_correct_candidate"] = correct.reindex(keys).fillna(False).to_numpy(dtype=bool)
    obs["candidate_miss"] = ~obs.has_correct_candidate
    return obs


def generate_trip_tables(folder: Path, trip_id: str, partition: str, *,
                         radius_m: float, max_candidates: int, crs: str,
                         include_reference: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict, float]:
    """Generate strict-labeled candidates plus one accounting row per raw GPS fix."""
    started = time.perf_counter()
    trip = load_trip(folder, trip_id=str(trip_id), require_route=include_reference,
                     include_route=include_reference)
    prepared = preprocess_trip(trip)
    track = prepared["track"].copy()
    road = build_road_network(prepared["nodes"], prepared["arcs"], crs=crs, build_graph=False)
    candidates = build_candidate_features(
        trip_id=str(trip_id), track_df=track, nodes_df=prepared["nodes"],
        arcs_df=prepared["arcs"], search_radius_m=radius_m,
        max_candidates=max_candidates, crs=crs, network=road)
    if not candidates.empty:
        candidates = add_relative_features(candidates)
        if include_reference:
            candidates = attach_nearest_route_labels(candidates, prepared["route"])
        else:
            candidates["label"] = np.nan
    else:
        candidates = pd.DataFrame(columns=["trip_id", "gps_point_id", "candidate_arc_id",
            "distance_m", "bearing_diff_deg", "speed_mps", "time_gap_s", "segment_length_m",
            "gps_lon", "gps_lat", "arc_bearing_deg", "gps_heading_deg", "n_candidates",
            "dist_rank", "dist_gap_m", "label"])
    ref_arc, ref_dist = _nearest_route_arcs(track, road, prepared.get("route"))
    observations = track[["gps_point_id", "timestamp_s", "lon", "lat", "coord_valid",
                          "time_gap_s", "distance_m", "speed_mps", "is_duplicate",
                          "implausible_speed"]].copy()
    observations.insert(0, "trip_id", str(trip_id))
    observations.insert(1, "partition", partition)
    observations["nearest_reference_arc_id"] = ref_arc
    observations["nearest_reference_distance_m"] = ref_dist
    observations = account_candidates(observations, candidates)
    return candidates, observations, prepared, road, time.perf_counter() - started


def filter_candidate_radius(candidates: pd.DataFrame, radius_m: float,
                            max_candidates: int, route: pd.DataFrame | None) -> pd.DataFrame:
    """Derive a radius/cap experiment from a max-radius candidate superset."""
    if candidates.empty:
        return candidates.copy()
    subset = candidates.loc[candidates.distance_m <= float(radius_m)].copy()
    subset = (subset.sort_values(["gps_point_id", "distance_m", "candidate_arc_id"])
              .groupby("gps_point_id", sort=False).head(int(max_candidates)).copy())
    subset = add_relative_features(subset)
    return attach_nearest_route_labels(subset, route)
