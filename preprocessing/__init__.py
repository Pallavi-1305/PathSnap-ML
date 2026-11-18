"""
preprocessing package – Member 1: Data Preprocessing and Candidate Generation
==============================================================================

Public API (import these from anywhere in the project):

    from preprocessing import (
        load_trip,
        preprocess_trip,
        generate_candidates,
        prepare_training_candidates,
    )

Each function is documented in its own module; see:
  - data_loader.py        → load_track / load_nodes / load_arcs / load_route / load_trip
  - preprocess.py         → preprocess_track / preprocess_trip
  - road_network.py       → build_road_network / calculate_distance / calculate_bearing
  - candidate_generation.py → generate_candidates / build_candidate_features /
                              attach_training_labels / prepare_training_candidates
"""

from .data_loader import (
    load_track,
    load_nodes,
    load_arcs,
    load_route,
    load_trip,
)

from .preprocess import (
    preprocess_track,
    preprocess_trip,
)

from .road_network import (
    build_road_network,
    calculate_distance,
    calculate_bearing,
)

from .candidate_generation import (
    generate_candidates,
    build_candidate_features,
    attach_training_labels,
    prepare_training_candidates,
)

__all__ = [
    # data loading
    "load_track",
    "load_nodes",
    "load_arcs",
    "load_route",
    "load_trip",
    # preprocessing
    "preprocess_track",
    "preprocess_trip",
    # road network
    "build_road_network",
    "calculate_distance",
    "calculate_bearing",
    # candidate generation
    "generate_candidates",
    "build_candidate_features",
    "attach_training_labels",
    "prepare_training_candidates",
]
