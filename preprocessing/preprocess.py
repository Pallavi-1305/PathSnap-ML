"""
preprocess.py – This is the main entry point for preprocessing. It wraps
======================================================================

Responsibility
--------------
Take the raw DataFrames produced by data_loader.py and clean / enrich them
ready for candidate generation.

What this module does
---------------------
  - Validates longitude / latitude ranges.
  - Flags (but does NOT silently drop) invalid or missing GPS observations
    so the caller can decide what to do.
  - Checks timestamp monotonicity.
  - Identifies duplicate GPS fixes.
  - Computes inter-observation time gaps (seconds).
  - Estimates travel speed from successive fixes.
  - Estimates GPS heading (bearing) from successive fixes.

What this module does NOT do
-----------------------------
  - It does not train any model.
  - It does not build the road network (see road_network.py).
  - It does not generate candidates (see candidate_generation.py).
  - It does not modify the .track, .nodes, .arcs, or .route files on disk.

GPS noise is expected.  The rule here is *flag, enrich, preserve* – do not
silently discard observations that might be recoverable.
"""

import logging
import math
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# WGS-84 bounding box for valid GPS coordinates.
LON_MIN, LON_MAX = -180.0, 180.0
LAT_MIN, LAT_MAX = -90.0, 90.0

# A speed above this value (m/s) is treated as physically implausible for a
# road vehicle.  600 km/h ≈ 167 m/s; we use a generous 200 m/s (~720 km/h).
MAX_PLAUSIBLE_SPEED_MPS = 200.0

# Earth's mean radius in metres (used for the Haversine formula).
EARTH_RADIUS_M = 6_371_000.0


# ---------------------------------------------------------------------------
# Internal geometry helpers
# ---------------------------------------------------------------------------

def _haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """
    Return the great-circle distance in metres between two WGS-84 points.

    Uses the Haversine formula.  Accurate to ~0.5 % for distances up to
    a few hundred kilometres.
    """
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    )
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(min(a, 1.0)))


def _bearing_deg(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """
    Return the initial bearing in degrees [0, 360) from point 1 to point 2.

    North = 0°, East = 90°, South = 180°, West = 270°.
    """
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlam = math.radians(lon2 - lon1)
    x = math.sin(dlam) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlam)
    bearing = math.degrees(math.atan2(x, y))
    return bearing % 360.0


# ---------------------------------------------------------------------------
# GPS track preprocessing
# ---------------------------------------------------------------------------

def preprocess_track(track_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and enrich a GPS track DataFrame (output of load_track).

    The function adds diagnostic columns and computed features.  It never
    removes rows from the DataFrame so that the gps_point_id indices remain
    stable and downstream modules can use them without offset correction.

    Parameters
    ----------
    track_df : pd.DataFrame
        Must contain columns: gps_point_id, lon, lat, timestamp_s.

    Returns
    -------
    pd.DataFrame – same number of rows as the input, with extra columns:

        lon_valid        bool   – True if longitude is in [-180, 180]
        lat_valid        bool   – True if latitude  is in [-90,  90]
        coord_valid      bool   – True if both lon and lat are valid
        is_null          bool   – True if lon or lat is NaN/None
        is_duplicate     bool   – True if this point is a spatial duplicate
                                  of the immediately preceding point
        time_gap_s       float  – seconds since the previous GPS fix
                                  (NaN for the first point)
        distance_m       float  – metres from the previous GPS fix
                                  (NaN for the first point or duplicates)
        speed_mps        float  – estimated speed in m/s
                                  (NaN when time_gap_s ≤ 0 or missing)
        heading_deg      float  – estimated travel heading in degrees [0,360)
                                  (NaN for the first point or when moving < 1m)
        implausible_speed bool  – True when speed_mps > MAX_PLAUSIBLE_SPEED_MPS
        ts_monotonic     bool   – True if this timestamp is ≥ the previous one

    Notes on design choices
    -----------------------
    - GPS noise is expected; we keep all points unless they are genuinely NaN.
    - 'is_duplicate' only checks the *immediately previous* point to avoid
      accidentally suppressing valid stop-and-go observations.
    - Heading is undefined for a stationary vehicle (< 1 m displacement).
    - speed_mps is undefined when time_gap_s ≤ 0 (simultaneous fixes).
    """
    required_cols = {"gps_point_id", "lon", "lat", "timestamp_s"}
    missing = required_cols - set(track_df.columns)
    if missing:
        raise ValueError(f"preprocess_track: missing columns {missing}")

    df = track_df.copy()

    # ------------------------------------------------------------------ #
    # 1. Null / missing checks
    # ------------------------------------------------------------------ #
    df["is_null"] = df["lon"].isna() | df["lat"].isna()
    null_count = df["is_null"].sum()
    if null_count:
        logger.warning("GPS track: %d point(s) have NaN coordinates.", null_count)

    # ------------------------------------------------------------------ #
    # 2. Coordinate range checks
    # ------------------------------------------------------------------ #
    df["lon_valid"] = df["lon"].between(LON_MIN, LON_MAX, inclusive="both")
    df["lat_valid"] = df["lat"].between(LAT_MIN, LAT_MAX, inclusive="both")
    df["coord_valid"] = df["lon_valid"] & df["lat_valid"] & ~df["is_null"]

    invalid_coords = (~df["coord_valid"]).sum()
    if invalid_coords:
        logger.warning(
            "GPS track: %d point(s) have invalid or out-of-range coordinates.",
            invalid_coords,
        )

    # ------------------------------------------------------------------ #
    # 3. Timestamp monotonicity
    # ------------------------------------------------------------------ #
    ts_prev = df["timestamp_s"].shift(1)
    df["ts_monotonic"] = df["timestamp_s"] >= ts_prev
    # First row has no previous timestamp → mark as True (not a problem).
    df.loc[df.index[0], "ts_monotonic"] = True

    non_mono = (~df["ts_monotonic"]).sum()
    if non_mono:
        logger.warning(
            "GPS track: %d timestamp(s) are not monotonically increasing.",
            non_mono,
        )

    # ------------------------------------------------------------------ #
    # 4. Time gap between consecutive fixes
    # ------------------------------------------------------------------ #
    df["time_gap_s"] = df["timestamp_s"].diff()  # NaN for row 0

    # ------------------------------------------------------------------ #
    # 5. Distance and speed between consecutive fixes
    #    (uses only valid, non-null points; others get NaN)
    # ------------------------------------------------------------------ #
    dist_values = [np.nan]  # first point has no predecessor
    heading_values = [np.nan]
    speed_values = [np.nan]

    for i in range(1, len(df)):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]

        # Both points must have valid coordinates.
        if prev["coord_valid"] and curr["coord_valid"]:
            d = _haversine_m(prev["lon"], prev["lat"], curr["lon"], curr["lat"])
            dist_values.append(d)

            dt = df.iloc[i]["time_gap_s"]
            if dt is not None and not np.isnan(dt) and dt > 0:
                speed_values.append(d / dt)
            else:
                speed_values.append(np.nan)

            if d >= 1.0:  # only compute heading if the vehicle moved ≥ 1 m
                heading_values.append(
                    _bearing_deg(prev["lon"], prev["lat"], curr["lon"], curr["lat"])
                )
            else:
                heading_values.append(np.nan)
        else:
            dist_values.append(np.nan)
            speed_values.append(np.nan)
            heading_values.append(np.nan)

    df["distance_m"] = dist_values
    df["speed_mps"] = speed_values
    df["heading_deg"] = heading_values

    # ------------------------------------------------------------------ #
    # 6. Duplicate detection (same spatial position as the preceding fix)
    # ------------------------------------------------------------------ #
    df["is_duplicate"] = df["distance_m"] == 0.0

    # ------------------------------------------------------------------ #
    # 7. Implausible speed flag
    # ------------------------------------------------------------------ #
    df["implausible_speed"] = df["speed_mps"] > MAX_PLAUSIBLE_SPEED_MPS

    implausible = df["implausible_speed"].sum()
    if implausible:
        logger.warning(
            "GPS track: %d fix(es) have implausible speed (> %.0f m/s).",
            implausible,
            MAX_PLAUSIBLE_SPEED_MPS,
        )

    # ------------------------------------------------------------------ #
    # 8. Summary log
    # ------------------------------------------------------------------ #
    logger.info(
        "preprocess_track complete: %d points | %d null | %d invalid_coord | "
        "%d non-monotonic ts | %d duplicates | %d implausible speed",
        len(df),
        null_count,
        invalid_coords,
        non_mono,
        df["is_duplicate"].sum(),
        implausible,
    )

    return df


# ---------------------------------------------------------------------------
# Combined trip preprocessor
# ---------------------------------------------------------------------------

def preprocess_trip(
    trip_data: Dict[str, object],
    drop_invalid_gps: bool = False,
) -> Dict[str, object]:
    """
    Clean and enrich all components of a loaded trip.

    This is the main entry point for preprocessing module preprocessing.  It wraps
    preprocess_track and validates the nodes/arcs DataFrames without
    modifying the original data.

    Parameters
    ----------
    trip_data : dict
        Output of data_loader.load_trip, containing keys:
          'trip_id', 'track', 'nodes', 'arcs', 'route' (may be None).

    drop_invalid_gps : bool, default False
        If True, rows where coord_valid is False are removed from the track
        before returning.  Set this only when you are certain you do not need
        those rows.  The default preserves all rows.

    Returns
    -------
    dict – same structure as the input but with an enriched 'track' DataFrame.

    Notes
    -----
    - gps_point_id values are reset to 0-based integers after any row removal
      so they remain contiguous.
    - nodes and arcs DataFrames are returned unchanged (validation warnings
      are logged if problems are found).
    """
    result = dict(trip_data)  # shallow copy; DataFrames are not copied here

    # --- GPS track ---
    track = preprocess_track(trip_data["track"])

    if drop_invalid_gps:
        n_before = len(track)
        track = track[track["coord_valid"]].copy()
        track["gps_point_id"] = np.arange(len(track), dtype=int)
        track = track.reset_index(drop=True)
        n_dropped = n_before - len(track)
        if n_dropped:
            logger.info(
                "Trip %s: dropped %d invalid GPS point(s).",
                trip_data["trip_id"],
                n_dropped,
            )

    result["track"] = track

    # --- Nodes sanity check ---
    nodes = trip_data["nodes"]
    null_nodes = nodes[["lon", "lat"]].isna().any(axis=1).sum()
    if null_nodes:
        logger.warning(
            "Trip %s: %d node(s) have NaN coordinates.",
            trip_data["trip_id"],
            null_nodes,
        )

    # --- Arcs sanity check ---
    arcs = trip_data["arcs"]
    max_node_id = len(nodes) - 1
    bad_from = arcs["from_node_id"].gt(max_node_id).sum()
    bad_to = arcs["to_node_id"].gt(max_node_id).sum()
    if bad_from or bad_to:
        logger.warning(
            "Trip %s: %d arc(s) have out-of-range from_node_id; "
            "%d arc(s) have out-of-range to_node_id.",
            trip_data["trip_id"],
            bad_from,
            bad_to,
        )

    return result


# ---------------------------------------------------------------------------
# Dataset summary generator
# ---------------------------------------------------------------------------

def summarise_dataset(
    trips: list[Dict],
) -> pd.DataFrame:
    """
    Generate a one-row-per-trip summary DataFrame from a list of preprocessed
    trip dicts.

    Parameters
    ----------
    trips : list of dicts
        Each dict is the output of preprocess_trip (or load_trip).

    Returns
    -------
    pd.DataFrame with columns:
        trip_id, n_gps_points, n_null_gps, n_invalid_gps,
        n_nodes, n_arcs, n_route_arcs (NaN when no route),
        min_timestamp_s, max_timestamp_s, duration_s
    """
    rows = []
    for t in trips:
        tid = t["trip_id"]
        track = t["track"]
        nodes = t["nodes"]
        arcs = t["arcs"]
        route = t.get("route")

        # Counts depend on whether preprocessing columns exist.
        n_null = int(track.get("is_null", pd.Series([False] * len(track))).sum())
        n_invalid = int(
            (~track.get("coord_valid", pd.Series([True] * len(track)))).sum()
        )

        rows.append(
            {
                "trip_id": tid,
                "n_gps_points": len(track),
                "n_null_gps": n_null,
                "n_invalid_gps": n_invalid,
                "n_nodes": len(nodes),
                "n_arcs": len(arcs),
                "n_route_arcs": len(route) if route is not None else np.nan,
                "min_timestamp_s": track["timestamp_s"].min(),
                "max_timestamp_s": track["timestamp_s"].max(),
                "duration_s": track["timestamp_s"].max()
                - track["timestamp_s"].min(),
            }
        )

    return pd.DataFrame(rows)
