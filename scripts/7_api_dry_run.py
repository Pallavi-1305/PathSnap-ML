"""Run and save an end-to-end API request for every student-test trip."""
from pathlib import Path
import sys

import pandas as pd
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.api import create_app


def main():
    client = TestClient(create_app(ROOT))
    trip_ids = client.get("/api/trips").json()["trip_ids"]
    if len(trip_ids) != 10:
        raise RuntimeError(f"Expected 10 student-test trips, found {len(trip_ids)}")
    rows = []
    for trip_id in trip_ids:
        response = client.post(f"/api/decode/{trip_id}")
        if response.status_code != 200:
            raise RuntimeError(f"API failed for {trip_id}: HTTP {response.status_code}: {response.text}")
        data = response.json()
        if data.get("reference_route_exposed") is not False:
            raise RuntimeError(f"API exposed reference route for {trip_id}")
        rows.append({"trip_id": trip_id, "http_status": response.status_code,
            "observation_count": data["observation_count"],
            "missing_candidate_observations": data["missing_candidate_observations"],
            "candidate_miss_rate": data["candidate_miss_rate"],
            "classifier_arc_f1": data["classifier_only_metrics"]["arc_f1"],
            "hmm_arc_f1": data["hmm_metrics"]["arc_f1"],
            "reference_route_exposed": data["reference_route_exposed"]})
    output = ROOT / "results/api_dry_run.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"API dry run passed: {len(rows)}/{len(trip_ids)} student trips; saved {output}")


if __name__ == "__main__":
    main()
