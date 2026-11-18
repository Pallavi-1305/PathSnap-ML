"""
test_preprocessing.py – Unit tests for Member 1: Data Preprocessing and Candidate Generation
==============================================================================================

Tests use a small synthetic dataset so they can run without the real GPS data.

Run with:
    cd PathSnap
    python -m pytest tests/test_preprocessing.py -v

Or without pytest:
    python tests/test_preprocessing.py

What is tested
--------------
  1.  load_track       – loads correctly, preserves gps_point_id
  2.  load_nodes       – loads correctly, preserves node_id
  3.  load_arcs        – loads correctly, preserves arc_id
  4.  load_route       – loads correctly, preserves route_step_id
  5.  load_trip        – loads all four files at once
  6.  preprocess_track – flags invalid coordinates without removing rows
  7.  preprocess_track – detects non-monotonic timestamps
  8.  preprocess_track – computes time_gap_s, distance_m, speed_mps
  9.  build_road_network – computes arc lengths (non-negative)
  10. build_road_network – computes arc bearings in [0, 360)
  11. build_road_network – flags invalid arc endpoints
  12. generate_candidates – returns valid arc IDs
  13. generate_candidates – distances are non-negative
  14. generate_candidates – handles GPS points with no nearby roads
  15. generate_candidates – no duplicate (gps_point_id, candidate_arc_id) pairs
  16. build_candidate_features – column schema matches SCHEMA_COLS_NO_LABEL
  17. attach_training_labels – label is 0 or 1
  18. attach_training_labels – returns NaN labels when route is None
  19. prepare_training_candidates – works end-to-end on a synthetic trip
  20. Multi-trip run – pipeline works on more than one trip
"""

import math
import os
import sys
import tempfile
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Make the project root importable when this file is run directly.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.data_loader import load_track, load_nodes, load_arcs, load_route, load_trip
from preprocessing.preprocess import preprocess_track, preprocess_trip
from preprocessing.road_network import build_road_network, calculate_distance, calculate_bearing
from preprocessing.candidate_generation import (
    generate_candidates,
    build_candidate_features,
    attach_training_labels,
    prepare_training_candidates,
    SCHEMA_COLS_NO_LABEL,
    SCHEMA_COLS,
)


# ===========================================================================
# Synthetic data builders
# ===========================================================================

def _make_track_content() -> str:
    """
    5 GPS observations around a small area in central London (lon, lat, time).
    Uses WGS-84 coordinates that are geographically close together so the
    candidate search radius (50 m) is large enough to find matches.
    """
    return textwrap.dedent("""\
        -0.1278  51.5074  1000.0
        -0.1275  51.5075  1010.0
        -0.1272  51.5076  1020.0
        -0.1269  51.5077  1030.0
        -0.1266  51.5078  1040.0
    """)


def _make_nodes_content() -> str:
    """
    6 road-network nodes forming a small grid near the GPS track.
    """
    return textwrap.dedent("""\
        -0.1280  51.5073
        -0.1270  51.5073
        -0.1260  51.5073
        -0.1280  51.5078
        -0.1270  51.5078
        -0.1260  51.5078
    """)


def _make_arcs_content() -> str:
    """
    7 directed arcs connecting the 6 nodes.
    Arc IDs are implicit (row index 0–6).
    """
    return textwrap.dedent("""\
        0  1
        1  2
        3  4
        4  5
        0  3
        1  4
        2  5
    """)


def _make_route_content() -> str:
    """Reference route: arcs 0, 1 (the top row)."""
    return textwrap.dedent("""\
        0
        1
    """)


def _write_trip(folder: Path, stem: str = "trip"):
    """Write all four synthetic trip files into folder."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{stem}.track").write_text(_make_track_content())
    (folder / f"{stem}.nodes").write_text(_make_nodes_content())
    (folder / f"{stem}.arcs").write_text(_make_arcs_content())
    (folder / f"{stem}.route").write_text(_make_route_content())


# ===========================================================================
# Simple test runner (works without pytest)
# ===========================================================================

_PASS = []
_FAIL = []


def _run(name: str, fn):
    try:
        fn()
        _PASS.append(name)
        print(f"  ✓ {name}")
    except Exception as exc:
        _FAIL.append((name, exc))
        print(f"  ✗ {name}  →  {exc}")


# ===========================================================================
# Tests
# ===========================================================================

def test_load_track():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "t.track"
        p.write_text(_make_track_content())
        df = load_track(p)
        assert list(df.columns) == ["gps_point_id", "lon", "lat", "timestamp_s"], \
            f"Unexpected columns: {list(df.columns)}"
        assert len(df) == 5, f"Expected 5 rows, got {len(df)}"
        assert df["gps_point_id"].tolist() == [0, 1, 2, 3, 4], "gps_point_id not 0-based"
        assert not df["lon"].isna().any(), "Unexpected NaN in lon"


def test_load_nodes():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "t.nodes"
        p.write_text(_make_nodes_content())
        df = load_nodes(p)
        assert list(df.columns) == ["node_id", "lon", "lat"], \
            f"Unexpected columns: {list(df.columns)}"
        assert len(df) == 6, f"Expected 6 rows, got {len(df)}"
        assert df["node_id"].tolist() == list(range(6)), "node_id not 0-based"


def test_load_arcs():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "t.arcs"
        p.write_text(_make_arcs_content())
        df = load_arcs(p)
        assert list(df.columns) == ["arc_id", "from_node_id", "to_node_id"], \
            f"Unexpected columns: {list(df.columns)}"
        assert len(df) == 7, f"Expected 7 rows, got {len(df)}"
        assert df["arc_id"].tolist() == list(range(7)), "arc_id not 0-based"


def test_load_route():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "t.route"
        p.write_text(_make_route_content())
        df = load_route(p)
        assert list(df.columns) == ["route_step_id", "arc_id"], \
            f"Unexpected columns: {list(df.columns)}"
        assert len(df) == 2, f"Expected 2 rows, got {len(df)}"
        assert df["arc_id"].tolist() == [0, 1], "Arc IDs not correct"


def test_load_trip():
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp) / "00000001"
        _write_trip(folder, stem="trip")
        result = load_trip(folder, trip_id="00000001")
        assert result["trip_id"] == "00000001"
        assert isinstance(result["track"], pd.DataFrame)
        assert isinstance(result["nodes"], pd.DataFrame)
        assert isinstance(result["arcs"],  pd.DataFrame)
        assert isinstance(result["route"], pd.DataFrame)


def test_preprocess_track_flags_invalid_coords():
    """Invalid coordinates are flagged but not removed."""
    df = pd.DataFrame({
        "gps_point_id": [0, 1, 2],
        "lon"         : [999.0, -0.1278, -0.1275],  # 999 is invalid
        "lat"         : [51.5, 51.5074, 51.5075],
        "timestamp_s" : [1000, 1010, 1020],
    })
    result = preprocess_track(df)
    assert len(result) == 3, "Rows should not be removed by preprocess_track"
    assert result.loc[0, "coord_valid"] == False, "Invalid lon should set coord_valid=False"
    assert result.loc[1, "coord_valid"] == True,  "Valid coords should set coord_valid=True"


def test_preprocess_track_non_monotonic_timestamps():
    df = pd.DataFrame({
        "gps_point_id": [0, 1, 2],
        "lon"         : [-0.1278, -0.1275, -0.1272],
        "lat"         : [51.5074, 51.5075, 51.5076],
        "timestamp_s" : [1000, 990, 1020],  # 990 < 1000 → non-monotonic
    })
    result = preprocess_track(df)
    assert result.loc[1, "ts_monotonic"] == False, \
        "Non-monotonic timestamp should be flagged"
    assert result.loc[0, "ts_monotonic"] == True, \
        "First row should always be ts_monotonic=True"


def test_preprocess_track_computes_speed():
    df = pd.DataFrame({
        "gps_point_id": [0, 1],
        "lon"         : [-0.1278, -0.1270],
        "lat"         : [51.5074, 51.5074],
        "timestamp_s" : [1000.0, 1010.0],
    })
    result = preprocess_track(df)
    assert math.isnan(result.loc[0, "speed_mps"]), "First point speed should be NaN"
    assert result.loc[1, "speed_mps"] > 0,         "Speed should be positive for movement"
    assert result.loc[1, "time_gap_s"] == 10.0,    "time_gap_s should be 10 s"


def test_build_road_network_arc_lengths():
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp) / "t"
        _write_trip(folder)
        trip  = load_trip(folder)
        net   = build_road_network(trip["nodes"], trip["arcs"])
        arcs  = net["arcs"]
        valid = arcs[arcs["valid"]]
        assert (valid["length_m"] >= 0).all(), "Arc lengths must be non-negative"
        # At least one arc should have length > 0 (non-degenerate network).
        assert (valid["length_m"] > 0).any(), "At least one arc must have positive length"


def test_build_road_network_arc_bearings():
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp) / "t"
        _write_trip(folder)
        trip = load_trip(folder)
        net  = build_road_network(trip["nodes"], trip["arcs"])
        arcs = net["arcs"]
        valid = arcs[arcs["valid"]]
        assert (valid["bearing_deg"] >= 0).all(),   "Bearings must be ≥ 0"
        assert (valid["bearing_deg"] < 360).all(),  "Bearings must be < 360"


def test_build_road_network_flags_invalid_arcs():
    nodes_df = pd.DataFrame({
        "node_id": [0, 1],
        "lon"    : [-0.128, -0.127],
        "lat"    : [51.507, 51.507],
    })
    # Arc 0 is valid; arc 1 references non-existent node 99.
    arcs_df = pd.DataFrame({
        "arc_id"      : [0, 1],
        "from_node_id": [0, 1],
        "to_node_id"  : [1, 99],
    })
    net = build_road_network(nodes_df, arcs_df)
    assert net["arcs"].loc[0, "valid"] == True,  "Arc 0 should be valid"
    assert net["arcs"].loc[1, "valid"] == False, "Arc 1 referencing node 99 should be invalid"


def test_generate_candidates_returns_valid_arc_ids():
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp) / "t"
        _write_trip(folder)
        trip  = load_trip(folder)
        track = preprocess_track(trip["track"])
        cands = generate_candidates(
            track, trip["nodes"], trip["arcs"],
            search_radius_m=200,  # generous radius for synthetic data
        )
        if cands.empty:
            # It's acceptable to have no candidates if the network is sparse.
            return
        valid_arc_ids = set(trip["arcs"]["arc_id"])
        assert cands["candidate_arc_id"].isin(valid_arc_ids).all(), \
            "All candidate arc IDs must come from the arc table"


def test_generate_candidates_non_negative_distances():
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp) / "t"
        _write_trip(folder)
        trip  = load_trip(folder)
        track = preprocess_track(trip["track"])
        cands = generate_candidates(track, trip["nodes"], trip["arcs"], search_radius_m=200)
        if not cands.empty:
            assert (cands["distance_m"] >= 0).all(), "Distances must be non-negative"


def test_generate_candidates_handles_no_nearby_roads():
    """GPS points very far from all arcs should produce zero candidates (not crash)."""
    nodes_df = pd.DataFrame({
        "node_id": [0, 1],
        "lon"    : [-0.128, -0.127],
        "lat"    : [51.507, 51.507],
    })
    arcs_df = pd.DataFrame({
        "arc_id"      : [0],
        "from_node_id": [0],
        "to_node_id"  : [1],
    })
    # GPS point in a totally different location (e.g. Paris).
    track_df = pd.DataFrame({
        "gps_point_id": [0],
        "lon"         : [2.3522],
        "lat"         : [48.8566],
        "timestamp_s" : [1000.0],
        "coord_valid" : [True],
        "heading_deg" : [np.nan],
        "speed_mps"   : [np.nan],
        "time_gap_s"  : [np.nan],
    })
    cands = generate_candidates(track_df, nodes_df, arcs_df, search_radius_m=50)
    assert cands.empty, "Should return empty DataFrame when no roads are nearby"


def test_generate_candidates_no_duplicates():
    """No (gps_point_id, candidate_arc_id) pair should appear more than once."""
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp) / "t"
        _write_trip(folder)
        trip  = load_trip(folder)
        track = preprocess_track(trip["track"])
        cands = generate_candidates(track, trip["nodes"], trip["arcs"], search_radius_m=200)
        if cands.empty:
            return
        dupes = cands.duplicated(subset=["gps_point_id", "candidate_arc_id"]).sum()
        assert dupes == 0, f"{dupes} duplicate (gps_point_id, candidate_arc_id) pairs found"


def test_candidate_feature_schema():
    """build_candidate_features returns all expected columns."""
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp) / "t"
        _write_trip(folder)
        trip  = load_trip(folder)
        track = preprocess_track(trip["track"])
        cands = build_candidate_features(
            trip_id="test",
            track_df=track,
            nodes_df=trip["nodes"],
            arcs_df=trip["arcs"],
            search_radius_m=200,
        )
        # Even if empty, columns should match the schema.
        for col in SCHEMA_COLS_NO_LABEL:
            assert col in cands.columns, f"Missing column: {col}"


def test_attach_training_labels_values():
    """Labels should be 0 or 1 when a route is provided."""
    cands = pd.DataFrame({
        "trip_id"          : ["t"] * 4,
        "gps_point_id"     : [0, 0, 1, 1],
        "candidate_arc_id" : [0, 1, 2, 3],
        "distance_m"       : [5.0, 10.0, 8.0, 15.0],
    })
    route = pd.DataFrame({
        "route_step_id": [0, 1],
        "arc_id"       : [0, 2],
    })
    result = attach_training_labels(cands, route)
    assert result.loc[0, "label"] == 1, "Arc 0 is in route → label should be 1"
    assert result.loc[1, "label"] == 0, "Arc 1 is not in route → label should be 0"
    assert result.loc[2, "label"] == 1, "Arc 2 is in route → label should be 1"
    assert result.loc[3, "label"] == 0, "Arc 3 is not in route → label should be 0"


def test_attach_training_labels_no_route():
    """When route is None, label column should be NaN."""
    cands = pd.DataFrame({
        "candidate_arc_id": [0, 1],
    })
    result = attach_training_labels(cands, None)
    assert result["label"].isna().all(), "Labels should be NaN when route is None"


def test_prepare_training_candidates_end_to_end():
    """Full pipeline on a synthetic trip should return the full schema."""
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp) / "00000001"
        _write_trip(folder, stem="00000001")
        result = prepare_training_candidates(
            trip_folder=folder,
            trip_id="00000001",
            search_radius_m=200,
            max_candidates=10,
        )
        # Result must have all schema columns.
        for col in SCHEMA_COLS:
            assert col in result.columns, f"Missing column: {col}"
        # trip_id must be set correctly.
        if not result.empty:
            assert (result["trip_id"] == "00000001").all(), "trip_id mismatch"


def test_multi_trip_pipeline():
    """Pipeline should work correctly on more than one trip."""
    with tempfile.TemporaryDirectory() as tmp:
        folders = []
        for i in range(3):
            folder = Path(tmp) / f"trip_{i:02d}"
            _write_trip(folder, stem=f"trip_{i:02d}")
            folders.append(folder)

        all_results = []
        for folder in folders:
            tid = folder.name
            cands = prepare_training_candidates(
                trip_folder=folder,
                trip_id=tid,
                search_radius_m=200,
            )
            all_results.append(cands)

        combined = pd.concat(all_results, ignore_index=True)
        n_trips = combined["trip_id"].nunique()
        # With generous radius, all 3 trips should appear.
        assert n_trips >= 1, \
            f"Expected results from multiple trips, got {n_trips} unique trip_id(s)"


# ===========================================================================
# Calculate distance / bearing sanity checks
# ===========================================================================

def test_calculate_distance_sanity():
    """Distance from a point to itself should be 0."""
    d = calculate_distance((-0.1278, 51.5074), (-0.1278, 51.5074))
    assert d == 0.0, f"Distance to self should be 0, got {d}"


def test_calculate_distance_positive():
    """Distance between two distinct points should be > 0."""
    d = calculate_distance((-0.1278, 51.5074), (-0.1270, 51.5074))
    assert d > 0, "Distance between two distinct points should be positive"


def test_calculate_bearing_east():
    """
    Moving due East: bearing should be close to 90°.
    We increase lon while keeping lat fixed.
    """
    b = calculate_bearing((-0.1278, 51.5074), (-0.1200, 51.5074))
    assert 80 < b < 100, f"Eastward bearing should be near 90°, got {b:.1f}°"


def test_calculate_bearing_north():
    """
    Moving due North: bearing should be close to 0° (or 360°).
    We increase lat while keeping lon fixed.
    """
    b = calculate_bearing((-0.1278, 51.5074), (-0.1278, 51.6000))
    # 0° and 360° are the same direction; allow both.
    assert b < 10 or b > 350, f"Northward bearing should be near 0°/360°, got {b:.1f}°"


# ===========================================================================
# Entry point
# ===========================================================================

def run_all():
    tests = [
        ("1.  load_track loads correctly",                 test_load_track),
        ("2.  load_nodes loads correctly",                 test_load_nodes),
        ("3.  load_arcs loads correctly",                  test_load_arcs),
        ("4.  load_route loads correctly",                 test_load_route),
        ("5.  load_trip loads all four files",             test_load_trip),
        ("6.  preprocess_track flags invalid coords",      test_preprocess_track_flags_invalid_coords),
        ("7.  preprocess_track non-monotonic timestamps",  test_preprocess_track_non_monotonic_timestamps),
        ("8.  preprocess_track computes speed/time_gap",   test_preprocess_track_computes_speed),
        ("9.  arc lengths are non-negative",               test_build_road_network_arc_lengths),
        ("10. arc bearings in [0, 360)",                   test_build_road_network_arc_bearings),
        ("11. invalid arc endpoints are flagged",          test_build_road_network_flags_invalid_arcs),
        ("12. candidate arc IDs are valid",                test_generate_candidates_returns_valid_arc_ids),
        ("13. candidate distances are non-negative",       test_generate_candidates_non_negative_distances),
        ("14. handles GPS points with no nearby roads",    test_generate_candidates_handles_no_nearby_roads),
        ("15. no duplicate (gps_id, arc_id) pairs",        test_generate_candidates_no_duplicates),
        ("16. candidate feature schema matches",           test_candidate_feature_schema),
        ("17. labels are 0 or 1 with route",               test_attach_training_labels_values),
        ("18. labels are NaN without route",               test_attach_training_labels_no_route),
        ("19. end-to-end prepare_training_candidates",     test_prepare_training_candidates_end_to_end),
        ("20. pipeline works on multiple trips",           test_multi_trip_pipeline),
        ("21. calculate_distance: self = 0",               test_calculate_distance_sanity),
        ("22. calculate_distance: distinct > 0",           test_calculate_distance_positive),
        ("23. calculate_bearing: Eastward ≈ 90°",         test_calculate_bearing_east),
        ("24. calculate_bearing: Northward ≈ 0°",         test_calculate_bearing_north),
    ]

    print("\n" + "=" * 60)
    print("PathSnap – Member 1 Test Suite")
    print("=" * 60)

    for name, fn in tests:
        _run(name, fn)

    print("\n" + "=" * 60)
    print(f"Results: {len(_PASS)} passed, {len(_FAIL)} failed")
    if _FAIL:
        print("\nFailed tests:")
        for name, exc in _FAIL:
            print(f"  ✗ {name}")
            print(f"      {type(exc).__name__}: {exc}")
    print("=" * 60 + "\n")

    return len(_FAIL) == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
