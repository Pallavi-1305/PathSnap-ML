"""
pathsnap/data.py
================
Parse and validate raw trips; project coordinates.
Responsibility: data in, clean arrays out.

This file re-exports preprocessing API so that the rest of the
pathsnap package (other project modules) can import from a single location:

    from pathsnap.data import load_trip, preprocess_trip, ...

All implementation lives in preprocessing/.
"""

# Re-export everything Member 1 implemented.
from preprocessing import (          # noqa: F401
    load_track,
    load_nodes,
    load_arcs,
    load_route,
    load_trip,
    preprocess_track,
    preprocess_trip,
    build_road_network,
    calculate_distance,
    calculate_bearing,
    generate_candidates,
    build_candidate_features,
    attach_training_labels,
    prepare_training_candidates,
)
from preprocessing.data_loader import list_trips                 # noqa: F401
from preprocessing.preprocess  import summarise_dataset          # noqa: F401
from preprocessing.candidate_generation import SCHEMA_COLS, SCHEMA_DTYPES  # noqa: F401
