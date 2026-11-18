"""
pathsnap/features.py
====================
Candidate generation per GPS point, feature engineering,
observation-level labels from .route.

This file re-exports the candidate generation API from Member 1 so that
model training and decoding modules can import from pathsnap.features directly.

    from pathsnap.features import (
        generate_candidates,
        build_candidate_features,
        attach_training_labels,
        prepare_training_candidates,
        SCHEMA_COLS,
        SCHEMA_DTYPES,
    )
"""

from preprocessing.candidate_generation import (    # noqa: F401
    generate_candidates,
    build_candidate_features,
    attach_training_labels,
    prepare_training_candidates,
    SCHEMA_COLS,
    SCHEMA_COLS_NO_LABEL,
    SCHEMA_DTYPES,
)
