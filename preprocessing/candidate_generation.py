"""
candidate_generation.py – Data Preprocessing and Candidate Generation
================================================================================

Responsibility
--------------
For every GPS observation, find all nearby road arcs (the *candidates*), then
compute a rich set of features for each (GPS point, candidate arc) pair.

This module also creates training labels from the reference .route file.

Output schema (candidate_features.csv)
----------------------------------------
  trip_id              str   – trip identifier (string, not a numeric feature)
  gps_point_id         int   – GPS observation index (0-based, not a numeric feature)
  candidate_arc_id     int   – arc index in the road network (not a numeric feature)
  distance_m           float – perpendicular distance from GPS point to arc (metres)
  bearing_diff_deg     float – |GPS heading − arc bearing| wrapped to [0, 180]
  speed_mps            float – estimated GPS travel speed at this observation
  time_gap_s           float – seconds since the previous GPS fix
  segment_length_m     float – length of the candidate road segment (metres)
  gps_lon              float – WGS-84 longitude of the GPS point
  gps_lat              float – WGS-84 latitude of the GPS point
  arc_bearing_deg      float – bearing of the candidate arc (degrees)
  gps_heading_deg      float – estimated travel heading at this GPS point (degrees)
  n_candidates         int   – number of candidates found for this GPS point
  label                int   – 1 if this arc is part of the reference route, else 0
                               (NaN when no reference route is available)

Note on IDs
-----------
trip_id, gps_point_id, and candidate_arc_id are identifiers, NOT numeric input
features for the ML model.  model training module should drop them before training.

Note on the search radius
--------------------------
The default search_radius_m=50 means we look for arcs whose *closest point*
is within 50 metres of each GPS fix.  For urban roads this is a generous but
appropriate radius.  For rural roads or low-accuracy GPS you may increase it.
If a GPS point has no candidates within the radius, it is flagged (a warning
is logged) and skipped in the output.

Note on labelling
-----------------
We cannot assume that GPS point i maps directly to route step i.
Instead we adopt a *set membership* approach:
  A candidate arc is labelled 1 if its arc_id appears anywhere in the
  reference route arc set for that trip.
This is conservative (every arc you ever drive over counts as correct),
which slightly over-labels but avoids false negatives.  Limitations are
documented in prepare_training_candidates.
"""

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .road_network import build_road_network, project_coords, DEFAULT_CRS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal geometry: point-to-segment distance
# ---------------------------------------------------------------------------

def _point_to_segment_distance(
    px: float, py: float,   # GPS point in metric coordinates
    ax: float, ay: float,   # arc start node
    bx: float, by: float,   # arc end node
) -> float:
    """
    Return the shortest (perpendicular) distance in metres from point P to
    the directed line segment A→B.

    If the foot of the perpendicular lies outside [A, B], the distance to
    the nearer endpoint is returned instead.

    All coordinates must be in the same metric CRS (e.g. EPSG:27700).
    """
    # Vector AB and AP
    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay

    ab_sq = abx * abx + aby * aby

    if ab_sq == 0.0:
        # Degenerate arc: start and end are the same node.
        return math.sqrt(apx * apx + apy * apy)

    # Parameter t represents how far along AB the foot of the perpendicular lies.
    t = max(0.0, min(1.0, (apx * abx + apy * aby) / ab_sq))

    # Closest point on segment to P.
    cx = ax + t * abx
    cy = ay + t * aby

    dx, dy = px - cx, py - cy
    return math.sqrt(dx * dx + dy * dy)


# ---------------------------------------------------------------------------
# Bearing difference helper
# ---------------------------------------------------------------------------

def _bearing_difference(b1: float, b2: float) -> float:
    """
    Return the absolute angular difference between two bearings in [0, 180].

    Both bearings are in degrees.  The result accounts for wrap-around so
    that e.g. the difference between 350° and 10° is 20°, not 340°.
    """
    diff = abs(b1 - b2) % 360.0
    return diff if diff <= 180.0 else 360.0 - diff


# ---------------------------------------------------------------------------
# Main candidate generation function
# ---------------------------------------------------------------------------

def generate_candidates(
    track_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    arcs_df: pd.DataFrame,
    search_radius_m: float = 50.0,
    max_candidates: int = 10,
    crs: str = DEFAULT_CRS,
    network: Optional[dict] = None,
) -> pd.DataFrame:
    """
    For every GPS observation, find all nearby road arcs and return a
    DataFrame with one row per (GPS point, candidate arc) pair.

    Parameters
    ----------
    track_df       : pd.DataFrame – output of preprocess_track.
                     Required columns: gps_point_id, lon, lat, heading_deg,
                     speed_mps, time_gap_s.
    nodes_df       : pd.DataFrame – output of load_nodes.
    arcs_df        : pd.DataFrame – output of load_arcs.
    search_radius_m: float – maximum perpendicular distance (metres) from a
                     GPS point to a candidate arc.  Default 50 m.
                     Tune this based on EDA: use a larger value if many GPS
                     points have zero candidates.
    max_candidates : int – maximum number of candidate arcs per GPS point.
                     When more than max_candidates arcs fall within the
                     radius, only the closest max_candidates are kept.
    crs            : str – EPSG code of the metric projection to use.
    network        : dict or None – pre-built road network (output of
                     build_road_network).  Pass this to avoid rebuilding
                     the network for every trip.

    Returns
    -------
    pd.DataFrame – the candidate feature table (see module docstring for the
                   full schema).  The 'label' column is NOT included here;
                   use attach_training_labels to add it.

    How the search works
    --------------------
    1.  Build (or reuse) the road network to get metric coordinates for
        every node and arc.
    2.  Project each GPS point to metric coordinates.
    3.  For each GPS point, iterate over all arcs and compute the
        perpendicular distance from the GPS point to that arc.
    4.  Keep arcs within search_radius_m, then sort by distance and keep
        the closest max_candidates.

    Performance note
    ----------------
    The naïve O(GPS_points × arcs) loop is acceptable for the dataset sizes
    described in the README (90 trips, each with a few hundred GPS points
    and a few thousand arcs).  For much larger networks, a spatial index
    (e.g. shapely STRtree or scipy.spatial.cKDTree) would speed this up.
    """
    # ------------------------------------------------------------------ #
    # 1. Build road network if not provided
    # ------------------------------------------------------------------ #
    if network is None:
        network = build_road_network(nodes_df, arcs_df, crs=crs)

    arcs_net  = network["arcs"]
    transformer = network.get("transformer")

    # Only consider arcs that have valid endpoints and non-NaN geometry.
    valid_arcs = arcs_net[arcs_net["valid"] & arcs_net["length_m"].notna()].copy()

    if valid_arcs.empty:
        logger.warning("generate_candidates: no valid arcs in road network.")
        return _empty_candidate_df()

    # Pre-extract arc arrays for fast NumPy access (avoids per-row DataFrame lookups).
    arc_ids       = valid_arcs["arc_id"].to_numpy(dtype=int)
    arc_from_x    = valid_arcs["from_x"].to_numpy(dtype=float)
    arc_from_y    = valid_arcs["from_y"].to_numpy(dtype=float)
    arc_to_x      = valid_arcs["to_x"].to_numpy(dtype=float)
    arc_to_y      = valid_arcs["to_y"].to_numpy(dtype=float)
    arc_lengths   = valid_arcs["length_m"].to_numpy(dtype=float)
    arc_bearings  = valid_arcs["bearing_deg"].to_numpy(dtype=float)

    # ------------------------------------------------------------------ #
    # 2. Project GPS points to metric CRS
    # ------------------------------------------------------------------ #
    gps_valid = track_df[track_df.get("coord_valid", pd.Series([True] * len(track_df)))].copy()
    if gps_valid.empty:
        logger.warning("generate_candidates: no valid GPS points in track.")
        return _empty_candidate_df()

    gps_x, gps_y = project_coords(
        gps_valid["lon"].to_numpy(),
        gps_valid["lat"].to_numpy(),
        target_crs=crs,
        transformer=transformer,
    )

    # ------------------------------------------------------------------ #
    # 3. Candidate search
    # ------------------------------------------------------------------ #
    seg_dx = arc_to_x - arc_from_x
    seg_dy = arc_to_y - arc_from_y
    seg_l2 = seg_dx * seg_dx + seg_dy * seg_dy
    seg_safe = np.where(seg_l2 == 0, 1.0, seg_l2)
    col_chunks = {c: [] for c in SCHEMA_COLS_NO_LABEL if c != "trip_id"}
    zero_candidate_points = 0

    # Spatial index on arc midpoints. A segment whose closest point is within r of a GPS point
    # has its midpoint within r + half its length, so this query returns every true candidate.
    mid_xy = np.column_stack([(arc_from_x + arc_to_x) / 2.0, (arc_from_y + arc_to_y) / 2.0])
    max_half_len = float(np.nanmax(arc_lengths)) / 2.0
    tree = cKDTree(mid_xy)
    cand_lists = tree.query_ball_point(
        np.column_stack([gps_x, gps_y]), r=search_radius_m + max_half_len, workers=-1
    )

    for k, (gps_row, px, py) in enumerate(zip(gps_valid.itertuples(index=False), gps_x, gps_y)):
        gps_id      = int(gps_row.gps_point_id)
        gps_lon     = float(gps_row.lon)
        gps_lat     = float(gps_row.lat)
        heading     = float(gps_row.heading_deg) if hasattr(gps_row, "heading_deg") and not _is_nan(gps_row.heading_deg) else np.nan
        speed_mps   = float(gps_row.speed_mps)  if hasattr(gps_row, "speed_mps")   and not _is_nan(gps_row.speed_mps)   else np.nan
        time_gap    = float(gps_row.time_gap_s)  if hasattr(gps_row, "time_gap_s")  and not _is_nan(gps_row.time_gap_s)  else np.nan

        # Vectorised point-to-segment distance to every valid arc (same geometry as
        # _point_to_segment_distance, computed in one NumPy pass per GPS point).
        cand = np.sort(np.asarray(cand_lists[k], dtype=np.int64))
        if cand.size == 0:
            zero_candidate_points += 1
            continue
        ax_, ay_ = arc_from_x[cand], arc_from_y[cand]
        dxs, dys, l2 = seg_dx[cand], seg_dy[cand], seg_l2[cand]
        t = np.clip(((px - ax_) * dxs + (py - ay_) * dys) / np.where(l2 == 0, 1.0, l2), 0.0, 1.0)
        t = np.where(l2 == 0, 0.0, t)
        dist_c = np.hypot(px - (ax_ + t * dxs), py - (ay_ + t * dys))

        within = dist_c <= search_radius_m
        if not within.any():
            zero_candidate_points += 1
            continue
        cw, dw = cand[within], dist_c[within]
        # default sort, same input order as the original, so ties break identically
        order = np.argsort(dw)[:max_candidates]
        sorted_idx = cw[order]
        distances_sel = dw[order]
        n_cands = len(sorted_idx)
        arc_bear = arc_bearings[sorted_idx]

        # Bearing difference wrapped to [0, 180]; NaN when the GPS heading is unknown.
        if not math.isnan(heading):
            bdiff = np.abs(heading - arc_bear) % 360.0
            bdiff = np.where(bdiff <= 180.0, bdiff, 360.0 - bdiff)
        else:
            bdiff = np.full(n_cands, np.nan)

        col_chunks["gps_point_id"].append(np.full(n_cands, gps_id, dtype=np.int64))
        col_chunks["candidate_arc_id"].append(arc_ids[sorted_idx].astype(np.int64))
        col_chunks["distance_m"].append(distances_sel)
        col_chunks["bearing_diff_deg"].append(bdiff)
        col_chunks["speed_mps"].append(np.full(n_cands, speed_mps))
        col_chunks["time_gap_s"].append(np.full(n_cands, time_gap))
        col_chunks["segment_length_m"].append(arc_lengths[sorted_idx])
        col_chunks["gps_lon"].append(np.full(n_cands, gps_lon))
        col_chunks["gps_lat"].append(np.full(n_cands, gps_lat))
        col_chunks["arc_bearing_deg"].append(arc_bear)
        col_chunks["gps_heading_deg"].append(np.full(n_cands, heading))
        col_chunks["n_candidates"].append(np.full(n_cands, n_cands, dtype=np.int64))

    if zero_candidate_points:
        logger.warning(
            "%d GPS point(s) had no candidate arcs within %.1f m radius. "
            "Consider increasing search_radius_m.",
            zero_candidate_points,
            search_radius_m,
        )

    if not col_chunks["gps_point_id"]:
        logger.warning("generate_candidates: no candidates found for any GPS point.")
        return _empty_candidate_df()

    # Build the table once from column arrays (much lower memory than one dict per row).
    result = pd.DataFrame({c: np.concatenate(v) for c, v in col_chunks.items()})
    logger.info(
        "generate_candidates: %d GPS points -> %d candidate rows (%d zero-candidate points).",
        len(gps_valid),
        len(result),
        zero_candidate_points,
    )
    return result


# ---------------------------------------------------------------------------
# Feature engineering wrapper
# ---------------------------------------------------------------------------

def build_candidate_features(
    trip_id: str,
    track_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    arcs_df: pd.DataFrame,
    search_radius_m: float = 50.0,
    max_candidates: int = 10,
    crs: str = DEFAULT_CRS,
    network: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Generate the full candidate feature table for one trip.

    This is a thin wrapper around generate_candidates that:
      - Adds the trip_id column.
      - Ensures the column ordering matches the agreed schema.

    Parameters
    ----------
    trip_id        : str – identifier for this trip.
    track_df       : preprocessed GPS track (output of preprocess_track).
    nodes_df, arcs_df : raw road-network tables (from load_nodes/load_arcs).
    search_radius_m, max_candidates, crs, network : forwarded to generate_candidates.

    Returns
    -------
    pd.DataFrame with columns in SCHEMA_COLS order (without 'label').
    """
    cands = generate_candidates(
        track_df=track_df,
        nodes_df=nodes_df,
        arcs_df=arcs_df,
        search_radius_m=search_radius_m,
        max_candidates=max_candidates,
        crs=crs,
        network=network,
    )

    if cands.empty:
        return _empty_candidate_df(trip_id=trip_id)

    cands.insert(0, "trip_id", trip_id)
    # Return in agreed column order.
    return cands[SCHEMA_COLS_NO_LABEL].copy()


# ---------------------------------------------------------------------------
# Training label attachment
# ---------------------------------------------------------------------------

def attach_training_labels(
    candidates_df: pd.DataFrame,
    route_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add a binary 'label' column to a candidate feature DataFrame.

    Labelling strategy: set membership
    ------------------------------------
    A candidate arc is labelled 1 if its arc_id appears anywhere in the
    reference route arc set for the trip.  Otherwise it is labelled 0.

    Why this approach?
    ------------------
    The GPS trajectory and the route do not share a common index.  The .track
    file has N GPS points; the .route file has M arcs.  N and M are not
    necessarily equal, and there is no guaranteed alignment between GPS point
    index i and route step i.

    Set-membership labelling avoids assuming any alignment.  It answers the
    question: "Is this candidate arc one that the vehicle ever drove over?"
    rather than "Is this candidate arc the one the vehicle was on at this
    exact GPS timestamp?"

    Limitations
    -----------
    - An arc that the vehicle passed over briefly (even at the very start or
      end of the trip) is labelled 1 for ALL GPS points that have it as a
      candidate, even if the vehicle was not on that arc at those timestamps.
    - If the reference route contains an arc that never appears as a candidate
      (e.g. because the GPS was far away), that arc simply does not appear in
      the output.
    - This may cause class imbalance if many arcs are shared across the route
      but are only valid candidates for a few GPS points.

    Parameters
    ----------
    candidates_df : pd.DataFrame – output of build_candidate_features.
    route_df      : pd.DataFrame – output of load_route.

    Returns
    -------
    pd.DataFrame – same as candidates_df with an integer 'label' column
                   appended (0 or 1).
    """
    if route_df is None or route_df.empty:
        logger.warning(
            "attach_training_labels: no route provided; label column will be NaN."
        )
        df = candidates_df.copy()
        df["label"] = np.nan
        return df

    route_arc_set = set(route_df["arc_id"].unique())
    df = candidates_df.copy()
    df["label"] = df["candidate_arc_id"].isin(route_arc_set).astype(int)

    n_pos = df["label"].sum()
    n_neg = len(df) - n_pos
    logger.info(
        "attach_training_labels: %d positive rows (%.1f %%), %d negative rows.",
        n_pos,
        100.0 * n_pos / max(len(df), 1),
        n_neg,
    )

    return df


# ---------------------------------------------------------------------------
# Top-level convenience function
# ---------------------------------------------------------------------------

def prepare_training_candidates(
    trip_folder,
    trip_id: Optional[str] = None,
    search_radius_m: float = 50.0,
    max_candidates: int = 10,
    crs: str = DEFAULT_CRS,
    drop_invalid_gps: bool = False,
) -> pd.DataFrame:
    """
    One-shot pipeline: load a training trip, preprocess it, generate candidates,
    and attach reference labels.

    This is the function that model training module should call when building the training
    dataset.

    Parameters
    ----------
    trip_folder     : str or Path – directory containing the trip files.
    trip_id         : str, optional – override auto-detected trip ID.
    search_radius_m : float – candidate search radius in metres.
    max_candidates  : int – max candidates per GPS point.
    crs             : str – metric CRS for geographic calculations.
    drop_invalid_gps: bool – if True, invalid GPS points are removed before
                      candidate generation.

    Returns
    -------
    pd.DataFrame – the labelled candidate feature table (full SCHEMA_COLS).

    Raises
    ------
    FileNotFoundError if the trip folder or required files do not exist.
    """
    # Import here to avoid circular imports.
    from .data_loader import load_trip
    from .preprocess import preprocess_trip

    # Load all files (require a .route file for training trips).
    trip_data = load_trip(trip_folder, trip_id=trip_id, require_route=True)
    effective_id = trip_data["trip_id"]

    # Preprocess GPS track.
    trip_data = preprocess_trip(trip_data, drop_invalid_gps=drop_invalid_gps)

    # Build road network once (reused inside build_candidate_features).
    network = build_road_network(
        trip_data["nodes"], trip_data["arcs"], crs=crs, build_graph=False
    )

    # Generate candidates + features.
    cands = build_candidate_features(
        trip_id=effective_id,
        track_df=trip_data["track"],
        nodes_df=trip_data["nodes"],
        arcs_df=trip_data["arcs"],
        search_radius_m=search_radius_m,
        max_candidates=max_candidates,
        crs=crs,
        network=network,
    )

    if cands.empty:
        logger.warning(
            "prepare_training_candidates: trip %s produced no candidates.",
            effective_id,
        )
        return _empty_candidate_df(trip_id=effective_id, with_label=True)

    # Attach reference labels.
    cands = add_relative_features(cands)
    cands = attach_nearest_route_labels(cands, trip_data["route"])

    return cands


# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

def add_relative_features(candidates_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per GPS point: dist_rank = rank of the candidate by distance (1 = closest),
    dist_gap_m = its distance minus the closest candidate's distance.
    Uses only candidates of the same GPS point, so it works the same at inference time.
    """
    df = candidates_df.copy()
    g = df.groupby("gps_point_id", sort=False)["distance_m"]
    df["dist_rank"] = g.rank(method="first").astype(float)
    df["dist_gap_m"] = df["distance_m"] - g.transform("min")
    return df


def attach_nearest_route_labels(candidates_df: pd.DataFrame, route_df) -> pd.DataFrame:
    """
    Stricter label: for each GPS point, label 1 only for the CLOSEST candidate whose arc
    is on the reference route; every other candidate is 0. A point with no on-route
    candidate gets no positive. At most one positive per GPS point.
    """
    df = candidates_df.copy()
    if route_df is None or route_df.empty:
        df["label"] = np.nan
        return df
    route_set = set(route_df["arc_id"].unique())
    on_route = np.flatnonzero(df["candidate_arc_id"].isin(route_set).to_numpy())
    if on_route.size:
        tmp = pd.DataFrame({
            "pid": df["gps_point_id"].to_numpy()[on_route],
            "d": df["distance_m"].to_numpy()[on_route],
            "row": on_route,
        })
        best = tmp.sort_values(["pid", "d", "row"]).groupby("pid", sort=False).head(1)["row"].to_numpy()
    else:
        best = np.array([], dtype=int)
    label = np.zeros(len(df), dtype=int)
    label[best] = 1
    df["label"] = label
    logger.info("attach_nearest_route_labels: %d positives over %d GPS points.",
                int(label.sum()), df["gps_point_id"].nunique())
    return df


#: Column order for the candidate feature table (without label).
SCHEMA_COLS_NO_LABEL = [
    "trip_id",
    "gps_point_id",
    "candidate_arc_id",
    "distance_m",
    "bearing_diff_deg",
    "speed_mps",
    "time_gap_s",
    "segment_length_m",
    "gps_lon",
    "gps_lat",
    "arc_bearing_deg",
    "gps_heading_deg",
    "n_candidates",
]

#: Full schema including the label column.
SCHEMA_COLS = SCHEMA_COLS_NO_LABEL + ["label"]

#: Data types for each column (for validation and documentation).
SCHEMA_DTYPES = {
    "trip_id"           : "object",   # string identifier
    "gps_point_id"      : "int64",    # integer identifier
    "candidate_arc_id"  : "int64",    # integer identifier
    "distance_m"        : "float64",  # ML feature
    "bearing_diff_deg"  : "float64",  # ML feature
    "speed_mps"         : "float64",  # ML feature
    "time_gap_s"        : "float64",  # ML feature
    "segment_length_m"  : "float64",  # ML feature
    "gps_lon"           : "float64",  # diagnostic / HMM
    "gps_lat"           : "float64",  # diagnostic / HMM
    "arc_bearing_deg"   : "float64",  # diagnostic / HMM
    "gps_heading_deg"   : "float64",  # diagnostic / HMM
    "n_candidates"      : "int64",    # diagnostic
    "label"             : "float64",  # 0 / 1 / NaN
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_nan(v) -> bool:
    """Return True when v is a float NaN (handles numpy scalars safely)."""
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return False


def _empty_candidate_df(
    trip_id: Optional[str] = None,
    with_label: bool = False,
) -> pd.DataFrame:
    """Return an empty DataFrame with the correct columns and types."""
    cols = SCHEMA_COLS if with_label else SCHEMA_COLS_NO_LABEL
    df = pd.DataFrame(columns=cols)
    for col, dtype in SCHEMA_DTYPES.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)
    return df
