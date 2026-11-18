"""Folium route maps with separate togglable GPS, candidates, predictions, and truth."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

try:
    import folium
except ImportError as exc:  # pragma: no cover - dependency message
    raise ImportError("Install the declared folium dependency to generate interactive route maps") from exc


def save_folium_route_map(track: pd.DataFrame, network: dict, candidates: pd.DataFrame,
                          classifier_route: list[int], hmm_route: list[int],
                          reference_route: list[int], out: Path, *, title: str,
                          missing_gps_ids: set[int] | None = None) -> None:
    valid_gps = track.dropna(subset=["lat", "lon"])
    center = [float(valid_gps.lat.mean()), float(valid_gps.lon.mean())] if len(valid_gps) else [0.0, 0.0]
    fmap = folium.Map(location=center, zoom_start=14, control_scale=True, tiles="OpenStreetMap")
    arcs = network["arcs"].set_index("arc_id")
    nodes = network["nodes"].set_index("node_id")
    layer_gps = folium.FeatureGroup(name="GPS observations", show=True)
    if len(valid_gps) > 1:
        folium.PolyLine(valid_gps[["lat", "lon"]].to_numpy().tolist(),
                        color="#222222", weight=2, opacity=.75, tooltip="GPS track").add_to(layer_gps)
    missing = missing_gps_ids or set()
    for row in valid_gps.itertuples(index=False):
        color = "red" if int(row.gps_point_id) in missing else "#222222"
        folium.CircleMarker([float(row.lat), float(row.lon)], radius=3,
                            color=color, fill=True, fill_opacity=.8,
                            tooltip=f"GPS point {row.gps_point_id}" + (" (no candidates)" if color == "red" else "")) .add_to(layer_gps)
    layer_gps.add_to(fmap)

    def arc_coords(arc_id: int):
        if int(arc_id) not in arcs.index:
            return None
        edge = arcs.loc[int(arc_id)]
        u, v = int(edge.from_node_id), int(edge.to_node_id)
        if u not in nodes.index or v not in nodes.index:
            return None
        a, b = nodes.loc[u], nodes.loc[v]
        return [[float(a.lat), float(a.lon)], [float(b.lat), float(b.lon)]]

    candidate_layer = folium.FeatureGroup(name="Candidate road segments", show=False)
    if not candidates.empty:
        for arc_id in sorted(set(candidates.candidate_arc_id.astype(int))):
            coords = arc_coords(arc_id)
            if coords:
                folium.PolyLine(coords, color="#b8b8b8", weight=2, opacity=.45,
                                tooltip=f"Candidate arc {arc_id}").add_to(candidate_layer)
    candidate_layer.add_to(fmap)

    def add_route(name: str, ids: list[int], color: str, weight: int, show: bool):
        layer = folium.FeatureGroup(name=name, show=show)
        for step, arc_id in enumerate(ids):
            coords = arc_coords(int(arc_id))
            if coords:
                folium.PolyLine(coords, color=color, weight=weight, opacity=.9,
                                tooltip=f"{name}, arc {arc_id}, step {step}").add_to(layer)
        layer.add_to(fmap)

    add_route("Classifier-only route", classifier_route, "#ff7f0e", 4, True)
    add_route("HMM + Viterbi route", hmm_route, "#1f77b4", 5, True)
    add_route("Reference route (evaluation only)", reference_route, "#2ca02c", 3, False)
    folium.LayerControl(collapsed=False).add_to(fmap)
    legend = ("<div style='position:fixed;bottom:28px;left:28px;z-index:9999;"
              "background:white;padding:10px;border:1px solid #777'>"
              f"<b>{title}</b><br>"
              "<span style='color:#222'>●</span> GPS observations &nbsp; "
              "<span style='color:red'>●</span> no candidate<br>"
              "<span style='color:#ff7f0e'>━</span> Classifier &nbsp; "
              "<span style='color:#1f77b4'>━</span> HMM &nbsp; "
              "<span style='color:#2ca02c'>━</span> Reference</div>")
    fmap.get_root().html.add_child(folium.Element(legend))
    out.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(out))
