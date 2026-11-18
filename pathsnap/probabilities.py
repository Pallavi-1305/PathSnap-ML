"""Build the label-free, identifier-keyed export consumed by Member 3."""
from __future__ import annotations

import numpy as np
import pandas as pd

KEYS = ["trip_id", "gps_point_id", "candidate_arc_id"]


def candidate_probability_export(candidates: pd.DataFrame, probabilities,
                                model_name: str) -> pd.DataFrame:
    missing = [column for column in KEYS if column not in candidates]
    if missing:
        raise ValueError(f"Candidate table is missing identifiers: {missing}")
    values = np.asarray(probabilities, dtype=float)
    if len(values) != len(candidates):
        raise ValueError("Probability count does not match candidate rows")
    exported = candidates[KEYS].copy()
    exported["trip_id"] = exported.trip_id.astype(str)
    exported["gps_point_id"] = exported.gps_point_id.astype(int)
    exported["candidate_arc_id"] = exported.candidate_arc_id.astype(int)
    if exported.duplicated(KEYS).any():
        raise ValueError("Candidate identifiers must be unique within trip and observation")
    exported["selected_model"] = str(model_name)
    exported["predicted_probability"] = np.clip(
        np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    return exported
