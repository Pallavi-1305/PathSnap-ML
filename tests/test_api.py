"""Frozen API route tests; the all-trip dry run is marked slow."""
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]


def test_api_lists_only_student_test_trips():
    from app.api import create_app

    client = TestClient(create_app(ROOT))
    response = client.get("/api/trips")
    assert response.status_code == 200
    listed = response.json()["trip_ids"]
    splits = pd.read_csv(ROOT / "data/splits.csv", dtype={"trip_id": str})
    expected = sorted(splits.loc[splits.partition == "student_test", "trip_id"].astype(str))
    assert listed == expected


@pytest.mark.slow
def test_api_frozen_pipeline_all_10_student_trips():
    from app.api import create_app

    artifact = ROOT / "artifacts/final_model.joblib"
    if not artifact.exists():
        pytest.skip("Run scripts/4_export_final.py to freeze the model first")
    client = TestClient(create_app(ROOT))
    ids = client.get("/api/trips").json()["trip_ids"]
    assert len(ids) == 10
    for trip_id in ids:
        response = client.post(f"/api/decode/{trip_id}")
        assert response.status_code == 200, f"{trip_id}: {response.text}"
        payload = response.json()
        assert payload["trip_id"] == trip_id
        assert payload["observation_count"] > 0
        assert payload["reference_route_exposed"] is False
        assert len(payload["gps_points"]) == payload["observation_count"]
        assert "classifier_only_metrics" in payload and "hmm_metrics" in payload
