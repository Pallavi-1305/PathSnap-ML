"""
road_network.py – Data Preprocessing and Candidate Generation
========================================================================

Responsibility
--------------
Take the nodes and arcs DataFrames (output of data_loader.py) and build
a richer road-network representation that includes:

  - Projected (metric) coordinates for every node.
  - Length and bearing for every arc.
  - A NetworkX directed graph for graph-theoretic queries (optional).

Why we project to metric coordinates
--------------------------------------
Raw longitude/latitude differences in degrees are NOT metres.  One degree of
latitude ≈ 111 km everywhere, but one degree of longitude ≈ 111 km × cos(lat),
which shrinks towards zero at the poles.

If we naïvely computed Euclidean distances in degree-space we would get
wildly wrong results, especially for North–South calculations.

We therefore use the pyproj library to project WGS-84 (EPSG:4326) to
British National Grid (EPSG:27700).  After projection every coordinate
is in metres, so ordinary Euclidean arithmetic is safe.

EPSG:27700 is the CRS specified in config.yaml for this dataset.  It is
appropriate for datasets covering Great Britain.  If your data are from
a different region, change the CRS to a suitable local projection (e.g.
EPSG:32633 for UTM zone 33N in central Europe).  The CRS is exposed as a
parameter so it can be changed without editing this file.
"""

import logging
import math
from typing import Optional, Tuple

import numpy as np
import pandas as pd

try:
    from pyproj import Transformer
    _PYPROJ_AVAILABLE = True
except ImportError:
    _PYPROJ_AVAILABLE = False

try:
    import networkx as nx
    _NX_AVAILABLE = True
except ImportError:
    _NX_AVAILABLE = False

logger = logging.getLogger(__name__)

# Default CRS for the project (British National Grid).
DEFAULT_CRS = "EPSG:27700"
WGS84_CRS   = "EPSG:4326"

# Earth's mean radius in metres (Haversine fallback).
EARTH_RADIUS_M = 6_371_000.0


# ---------------------------------------------------------------------------
# Low-level geometry helpers (public so HMM decoding module can import them)
# ---------------------------------------------------------------------------

def calculate_distance(point1: Tuple[float, float],
                        point2: Tuple[float, float],
                        projected: bool = False) -> float:
    """
    Calculate the distance in metres between two geographic points.

    Parameters
    ----------
    point1, point2 : (x, y) tuples.
        When projected=False  → (longitude, latitude) in decimal degrees (WGS-84).
        When projected=True   → (easting, northing) in metres in the project CRS.

    projected : bool, default False
        Set to True when the coordinates are already in a metric projection.
        In that case the function uses plain Euclidean distance.

    Returns
    -------
    float – distance in metres.
    """
    if projected:
        # Already in metres; simple Euclidean.
        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]
        return math.sqrt(dx * dx + dy * dy)

    # Haversine formula for WGS-84 lon/lat.
    lon1, lat1 = point1
    lon2, lat2 = point2
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    )
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(min(a, 1.0)))


def calculate_bearing(point1: Tuple[float, float],
                       point2: Tuple[float, float],
                       projected: bool = False) -> float:
    """
    Calculate the initial bearing (azimuth) in degrees from point1 to point2.

    Returns a value in [0, 360) where 0° = North, 90° = East, etc.

    Parameters
    ----------
    point1, point2 : (x, y) tuples.
        When projected=False  → (longitude, latitude) in WGS-84 degrees.
        When projected=True   → (easting, northing) in metric CRS.

    projected : bool, default False
        When True, bearing is computed using simple 2-D atan2 on metric coords.

    Returns
    -------
    float – bearing in degrees [0, 360).
    """
    if projected:
        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]
        bearing = math.degrees(math.atan2(dx, dy))  # note: (dx, dy) for North-up
        return bearing % 360.0

    # Spherical bearing from WGS-84 lon/lat.
    lon1, lat1 = point1
    lon2, lat2 = point2
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlam = math.radians(lon2 - lon1)
    x = math.sin(dlam) * math.cos(phi2)
    y = (
        math.cos(phi1) * math.sin(phi2)
        - math.sin(phi1) * math.cos(phi2) * math.cos(dlam)
    )
    bearing = math.degrees(math.atan2(x, y))
    return bearing % 360.0


# ---------------------------------------------------------------------------
# CRS projection helper
# ---------------------------------------------------------------------------

def _build_transformer(target_crs: str = DEFAULT_CRS):
    """
    Build a pyproj Transformer from WGS-84 to target_crs.

    Returns None if pyproj is not installed (Haversine fallback will be used).
    """
    if not _PYPROJ_AVAILABLE:
        logger.warning(
            "pyproj is not installed.  Geographic distances will fall back to "
            "the Haversine formula.  Install pyproj for metric projections: "
            "  pip install pyproj"
        )
        return None

    try:
        return Transformer.from_crs(WGS84_CRS, target_crs, always_xy=True)
    except Exception as exc:
        logger.warning("Could not build pyproj Transformer: %s", exc)
        return None


def project_coords(
    lon: np.ndarray,
    lat: np.ndarray,
    target_crs: str = DEFAULT_CRS,
    transformer=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project WGS-84 lon/lat arrays to a metric CRS.

    Parameters
    ----------
    lon, lat      : 1-D NumPy arrays in WGS-84 degrees.
    target_crs    : EPSG string for the target CRS.
    transformer   : pre-built pyproj Transformer (or None → one is created).

    Returns
    -------
    (x, y) : 1-D NumPy arrays in metres in target_crs.

    When pyproj is unavailable, the function returns a *flat-Earth
    approximation* (lon scaled by cos(mean_lat) × 111_320, lat × 111_320).
    This is good to ~1 % within a 50 km region and is acceptable for the
    candidate-search radius calculation.
    """
    if transformer is None:
        transformer = _build_transformer(target_crs)

    if transformer is not None:
        x, y = transformer.transform(lon, lat)
        return np.asarray(x), np.asarray(y)

    # Flat-Earth approximation fallback.
    mean_lat_rad = math.radians(float(np.nanmean(lat)))
    x = np.asarray(lon, dtype=float) * math.cos(mean_lat_rad) * 111_320.0
    y = np.asarray(lat, dtype=float) * 111_320.0
    return x, y


# ---------------------------------------------------------------------------
# Road-network builder
# ---------------------------------------------------------------------------

def build_road_network(
    nodes_df: pd.DataFrame,
    arcs_df: pd.DataFrame,
    crs: str = DEFAULT_CRS,
    build_graph: bool = True,
) -> dict:
    """
    Build an enriched road-network representation from nodes and arcs tables.

    Parameters
    ----------
    nodes_df : pd.DataFrame
        Output of load_nodes; columns: node_id, lon, lat.
    arcs_df  : pd.DataFrame
        Output of load_arcs;  columns: arc_id, from_node_id, to_node_id.
    crs      : str
        EPSG code of the metric projection to use.  Default: EPSG:27700.

    Returns
    -------
    dict with keys:

        'nodes'    : pd.DataFrame – nodes with added columns:
                       x (easting metres), y (northing metres)

        'arcs'     : pd.DataFrame – arcs with added columns:
                       from_x, from_y, to_x, to_y (endpoint coordinates, m)
                       length_m (arc length in metres)
                       bearing_deg (direction of travel, degrees [0,360))
                       valid (True when both endpoints reference valid nodes)

        'graph'    : networkx.DiGraph or None
                       Directed graph; edge attributes include arc_id,
                       length_m, bearing_deg.
                       None when networkx is not installed.

        'transformer' : pyproj.Transformer or None
                        Pre-built transformer for reuse (e.g. projecting GPS
                        points later); None when pyproj is not installed.

        'crs'      : str – CRS string used for projection.

    Design notes
    ------------
    - arc_id and node_id are preserved as-is from the input DataFrames.
    - Arcs whose from_node_id or to_node_id references a non-existent node
      are flagged with valid=False; they are retained in the DataFrame so
      that arc_id indexing stays intact.
    - Length and bearing are computed using metric projected coordinates
      (Euclidean on projected coords; accurate for local road networks).
    """
    # ------------------------------------------------------------------ #
    # 1. Project nodes to metric CRS
    # ------------------------------------------------------------------ #
    transformer = _build_transformer(crs)
    x_arr, y_arr = project_coords(
        nodes_df["lon"].to_numpy(),
        nodes_df["lat"].to_numpy(),
        target_crs=crs,
        transformer=transformer,
    )

    nodes = nodes_df.copy()
    nodes["x"] = x_arr
    nodes["y"] = y_arr

    # Build a fast lookup: node_id → (x, y)
    # (Using .set_index().to_dict() is faster than row iteration for large networks.)
    node_x = nodes.set_index("node_id")["x"].to_dict()
    node_y = nodes.set_index("node_id")["y"].to_dict()
    valid_node_ids = set(nodes["node_id"])

    # ------------------------------------------------------------------ #
    # 2. Enrich arcs
    # ------------------------------------------------------------------ #
    arcs = arcs_df.copy()

    # Validity flag: both endpoints must reference known node IDs.
    arcs["valid"] = (
        arcs["from_node_id"].isin(valid_node_ids)
        & arcs["to_node_id"].isin(valid_node_ids)
    )

    invalid_arcs = (~arcs["valid"]).sum()
    if invalid_arcs:
        logger.warning(
            "%d arc(s) reference unknown node IDs and will be flagged invalid.",
            invalid_arcs,
        )

    # Map node coordinates to each arc endpoint.
    arcs["from_x"] = arcs["from_node_id"].map(node_x)
    arcs["from_y"] = arcs["from_node_id"].map(node_y)
    arcs["to_x"]   = arcs["to_node_id"].map(node_x)
    arcs["to_y"]   = arcs["to_node_id"].map(node_y)

    # Compute arc lengths and bearings using metric (projected) coordinates.
    dx = arcs["to_x"] - arcs["from_x"]
    dy = arcs["to_y"] - arcs["from_y"]

    arcs["length_m"]    = np.sqrt(dx**2 + dy**2)
    # atan2(dx, dy) gives North-up azimuth (North = 0°, East = 90°).
    arcs["bearing_deg"] = np.degrees(np.arctan2(dx, dy)) % 360.0

    # Set length and bearing to NaN for invalid arcs.
    arcs.loc[~arcs["valid"], ["length_m", "bearing_deg"]] = np.nan

    logger.info(
        "build_road_network: %d nodes, %d arcs (%d invalid), CRS=%s",
        len(nodes),
        len(arcs),
        invalid_arcs,
        crs,
    )

    # ------------------------------------------------------------------ #
    # 3. Build NetworkX directed graph (optional)
    # ------------------------------------------------------------------ #
    graph = None
    if _NX_AVAILABLE and build_graph:
        graph = nx.DiGraph()

        # Add nodes with their projected coordinates as attributes.
        for _, row in nodes.iterrows():
            graph.add_node(
                int(row["node_id"]),
                lon=row["lon"],
                lat=row["lat"],
                x=row["x"],
                y=row["y"],
            )

        # Add directed edges for valid arcs only.
        valid_arcs = arcs[arcs["valid"]]
        for _, row in valid_arcs.iterrows():
            graph.add_edge(
                int(row["from_node_id"]),
                int(row["to_node_id"]),
                arc_id=int(row["arc_id"]),
                length_m=float(row["length_m"]),
                bearing_deg=float(row["bearing_deg"]),
            )

        logger.debug("NetworkX DiGraph: %d nodes, %d edges", graph.number_of_nodes(), graph.number_of_edges())
    else:
        logger.info(
            "networkx is not installed.  Graph will be None.  "
            "Install it with: pip install networkx"
        )

    return {
        "nodes": nodes,
        "arcs": arcs,
        "graph": graph,
        "transformer": transformer,
        "crs": crs,
    }


