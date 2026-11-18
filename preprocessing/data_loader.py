"""
data_loader.py – Data Preprocessing and Candidate Generation
=======================================================================

Responsibility
--------------
Read the four raw trip files (.track, .nodes, .arcs, .route) from disk and
return clean, well-typed Pandas DataFrames with meaningful column names.

None of the functions here modify the original files; they are read-only.

File formats (all whitespace-separated, no header row)
-------------------------------------------------------
  .track  – lon  lat  timestamp_s          (float float float)
  .nodes  – lon  lat                        (float float)
  .arcs   – from_node_id  to_node_id        (int   int)
  .route  – arc_id                          (int)

Node IDs and arc IDs are zero-based row indices in their respective files.
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Individual file loaders
# ---------------------------------------------------------------------------

def load_track(file_path: str | Path) -> pd.DataFrame:
    """
    Load a `.track` file (GPS trajectory observations).

    Each row represents one GPS fix recorded during a trip.

    Parameters
    ----------
    file_path : str or Path
        Full path to the .track file.

    Returns
    -------
    pd.DataFrame with columns:
        gps_point_id  – integer index (0-based row number)
        lon           – WGS-84 longitude  (float, degrees)
        lat           – WGS-84 latitude   (float, degrees)
        timestamp_s   – UNIX-style timestamp in seconds (float)

    Raises
    ------
    FileNotFoundError  if the file does not exist.
    ValueError         if the file has fewer than 3 columns.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Track file not found: {file_path}")

    # Read as generic whitespace-delimited data with no header.
    # 'sep=None' + 'engine="python"' auto-detects any whitespace.
    try:
        df = pd.read_csv(
            file_path,
            sep=r"\s+",
            header=None,
            engine="python",
            dtype=float,
            na_values=["", "NA", "NaN", "nan", "null"],
        )
    except Exception as exc:
        raise ValueError(f"Could not parse {file_path}: {exc}") from exc

    if df.shape[1] < 3:
        raise ValueError(
            f"{file_path} has only {df.shape[1]} columns; expected at least 3 "
            "(lon, lat, timestamp_s)."
        )

    # Keep only the first three columns in case extra columns exist.
    df = df.iloc[:, :3].copy()
    df.columns = ["lon", "lat", "timestamp_s"]

    # Attach the GPS point ID as the first column (0-based row index).
    df.insert(0, "gps_point_id", np.arange(len(df), dtype=int))

    logger.debug("Loaded %d GPS points from %s", len(df), file_path.name)
    return df


def load_nodes(file_path: str | Path) -> pd.DataFrame:
    """
    Load a `.nodes` file (road-network node coordinates).

    Parameters
    ----------
    file_path : str or Path
        Full path to the .nodes file.

    Returns
    -------
    pd.DataFrame with columns:
        node_id  – integer index (0-based row number)
        lon      – WGS-84 longitude  (float, degrees)
        lat      – WGS-84 latitude   (float, degrees)

    Raises
    ------
    FileNotFoundError  if the file does not exist.
    ValueError         if the file has fewer than 2 columns.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Nodes file not found: {file_path}")

    try:
        df = pd.read_csv(
            file_path,
            sep=r"\s+",
            header=None,
            engine="python",
            dtype=float,
            na_values=["", "NA", "NaN", "nan", "null"],
        )
    except Exception as exc:
        raise ValueError(f"Could not parse {file_path}: {exc}") from exc

    if df.shape[1] < 2:
        raise ValueError(
            f"{file_path} has only {df.shape[1]} columns; expected at least 2 (lon, lat)."
        )

    df = df.iloc[:, :2].copy()
    df.columns = ["lon", "lat"]
    df.insert(0, "node_id", np.arange(len(df), dtype=int))

    logger.debug("Loaded %d nodes from %s", len(df), file_path.name)
    return df


def load_arcs(file_path: str | Path) -> pd.DataFrame:
    """
    Load an `.arcs` file (road-network directed edges).

    Each row represents one directed road segment connecting two network nodes.

    Parameters
    ----------
    file_path : str or Path
        Full path to the .arcs file.

    Returns
    -------
    pd.DataFrame with columns:
        arc_id       – integer index (0-based row number)
        from_node_id – start node of the arc (int)
        to_node_id   – end node of the arc   (int)

    Raises
    ------
    FileNotFoundError  if the file does not exist.
    ValueError         if the file has fewer than 2 columns.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Arcs file not found: {file_path}")

    try:
        df = pd.read_csv(
            file_path,
            sep=r"\s+",
            header=None,
            engine="python",
            dtype=int,
            na_values=["", "NA", "NaN", "nan", "null"],
        )
    except Exception as exc:
        raise ValueError(f"Could not parse {file_path}: {exc}") from exc

    if df.shape[1] < 2:
        raise ValueError(
            f"{file_path} has only {df.shape[1]} columns; "
            "expected at least 2 (from_node_id, to_node_id)."
        )

    df = df.iloc[:, :2].copy()
    df.columns = ["from_node_id", "to_node_id"]
    df.insert(0, "arc_id", np.arange(len(df), dtype=int))

    logger.debug("Loaded %d arcs from %s", len(df), file_path.name)
    return df


def load_route(file_path: str | Path) -> pd.DataFrame:
    """
    Load a `.route` file (reference route arc sequence).

    The route lists the arc IDs that the vehicle actually drove, in order.
    This is used *only* during training to generate labels – it is never
    available at prediction time.

    Parameters
    ----------
    file_path : str or Path
        Full path to the .route file.

    Returns
    -------
    pd.DataFrame with columns:
        route_step_id – step index along the route (0-based)
        arc_id        – the arc ID driven at this step (int)

    Raises
    ------
    FileNotFoundError  if the file does not exist.
    ValueError         if the file is empty or malformed.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Route file not found: {file_path}")

    try:
        df = pd.read_csv(
            file_path,
            sep=r"\s+",
            header=None,
            engine="python",
            dtype=int,
            na_values=["", "NA", "NaN", "nan", "null"],
        )
    except Exception as exc:
        raise ValueError(f"Could not parse {file_path}: {exc}") from exc

    if df.empty:
        raise ValueError(f"{file_path} is empty; cannot load route.")

    # Only the first column matters; ignore extras.
    df = df.iloc[:, :1].copy()
    df.columns = ["arc_id"]
    df.insert(0, "route_step_id", np.arange(len(df), dtype=int))

    logger.debug("Loaded %d route arcs from %s", len(df), file_path.name)
    return df


# ---------------------------------------------------------------------------
# Combined trip loader
# ---------------------------------------------------------------------------

def load_trip(
    trip_folder: str | Path,
    trip_id: Optional[str] = None,
    require_route: bool = False,
    include_route: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Load all files belonging to one trip folder.

    A trip folder contains files with a common stem name and four extensions:
      <stem>.track   <stem>.nodes   <stem>.arcs   [<stem>.route]

    The function discovers the stem automatically by looking for a .track file.

    Parameters
    ----------
    trip_folder   : str or Path  – directory containing the trip files.
    trip_id       : str, optional – override the trip identifier in the output;
                                    defaults to the stem of the trip files.
    require_route : bool – if True, raise an error when no .route file exists;
                           if False (default), the 'route' key is None when
                           no .route file is found (e.g. evaluation trips).

    Returns
    -------
    dict with keys:
        'trip_id'  : str
        'track'    : pd.DataFrame   (from load_track)
        'nodes'    : pd.DataFrame   (from load_nodes)
        'arcs'     : pd.DataFrame   (from load_arcs)
        'route'    : pd.DataFrame or None

    Raises
    ------
    FileNotFoundError  if the folder does not exist or no .track file is found.
    FileNotFoundError  if require_route=True and no .route file exists.
    """
    trip_folder = Path(trip_folder)
    if not trip_folder.is_dir():
        raise FileNotFoundError(f"Trip folder not found: {trip_folder}")

    # Find the common stem by looking for a .track file.
    track_files = list(trip_folder.glob("*.track"))
    if not track_files:
        raise FileNotFoundError(
            f"No .track file found in {trip_folder}. "
            "Make sure the trip data has been placed in this directory."
        )

    # Use the first .track file found (there should be exactly one per folder).
    stem = track_files[0].stem
    if len(track_files) > 1:
        logger.warning(
            "Multiple .track files found in %s; using %s", trip_folder, stem
        )

    # Determine the trip identifier.
    effective_trip_id = trip_id if trip_id is not None else stem

    # Load mandatory files.
    track_df = load_track(trip_folder / f"{stem}.track")
    nodes_df = load_nodes(trip_folder / f"{stem}.nodes")
    arcs_df = load_arcs(trip_folder / f"{stem}.arcs")

    # Load optional .route file only for training/evaluation callers that ask.
    route_path = trip_folder / f"{stem}.route"
    if not include_route:
        route_df = None
    elif route_path.exists():
        route_df = load_route(route_path)
    elif require_route:
        raise FileNotFoundError(
            f"No .route file found at {route_path} and require_route=True."
        )
    else:
        route_df = None
        logger.info(
            "No .route file for trip %s (expected for evaluation trips).",
            effective_trip_id,
        )

    logger.info(
        "Loaded trip %s: %d GPS points, %d nodes, %d arcs, route=%s",
        effective_trip_id,
        len(track_df),
        len(nodes_df),
        len(arcs_df),
        "yes" if route_df is not None else "no",
    )

    return {
        "trip_id": effective_trip_id,
        "track": track_df,
        "nodes": nodes_df,
        "arcs": arcs_df,
        "route": route_df,
    }


# ---------------------------------------------------------------------------
# Dataset inventory helper
# ---------------------------------------------------------------------------

def list_trips(data_dir: str | Path) -> pd.DataFrame:
    """
    Scan a directory tree and return an inventory of all discoverable trips.

    The function looks for sub-directories that contain at least one .track
    file (one subdirectory = one trip).

    Parameters
    ----------
    data_dir : str or Path – root of the data directory (e.g. 'data/train').

    Returns
    -------
    pd.DataFrame with columns:
        trip_id    – stem of the trip files
        folder     – absolute path to the trip folder
        has_route  – whether a .route file exists (True/False)
    """
    data_dir = Path(data_dir)
    rows = []

    # Walk through all immediate subdirectories.
    for sub in sorted(data_dir.iterdir()):
        if not sub.is_dir():
            continue
        track_files = list(sub.glob("*.track"))
        if not track_files:
            continue
        stem = track_files[0].stem
        rows.append(
            {
                "trip_id": stem,
                "folder": str(sub.resolve()),
                "has_route": (sub / f"{stem}.route").exists(),
            }
        )

    if not rows:
        logger.warning("No trips found under %s", data_dir)

    return pd.DataFrame(rows, columns=["trip_id", "folder", "has_route"])
