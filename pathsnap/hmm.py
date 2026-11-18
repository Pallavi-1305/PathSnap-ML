"""Directed road-network HMM and Viterbi map matcher.

Classifier probabilities form the emissions: log(max(floor, P(arc is correct | features))).
Distance is already one classifier feature, so no second distance Gaussian is added.
Transitions use directed network distance and GPS displacement; reverse links must exist.
"""
from __future__ import annotations
from dataclasses import dataclass
import heapq
import math
from typing import Iterable, Mapping, Sequence
import pandas as pd

@dataclass(frozen=True)
class Transition:
    log_probability: float
    reachable: bool
    network_distance_m: float = math.inf

class DirectedRoadNetwork:
    """Directed edge graph built from Member 1's .arcs row-index convention."""
    def __init__(self, arcs: pd.DataFrame, lengths: Mapping[int, float] | None = None):
        required = {"arc_id", "from_node_id", "to_node_id"}
        if not required.issubset(arcs.columns):
            raise ValueError(f"arcs is missing {sorted(required - set(arcs.columns))}")
        self.arc, self.edge_lengths = {}, {}
        self.adjacency: dict[int, list[tuple[int, float]]] = {}
        for row in arcs.itertuples(index=False):
            aid, u, v = int(row.arc_id), int(row.from_node_id), int(row.to_node_id)
            length = float((lengths or {}).get(aid, 0.0))
            if not math.isfinite(length) or length < 0:
                length = 0.0
            self.arc[aid] = (u, v)
            self.edge_lengths[aid] = length
            self.adjacency.setdefault(u, []).append((v, length))
        self._dist_cache: dict[tuple[int, float], dict[int, float]] = {}

    def shortest_from(self, source: int, max_distance_m: float) -> dict[int, float]:
        key = (source, float(max_distance_m))
        if key not in self._dist_cache:
            dist, queue = {source: 0.0}, [(0.0, source)]
            while queue:
                cost, node = heapq.heappop(queue)
                if cost != dist[node]:
                    continue
                if cost > max_distance_m:
                    break
                for nxt, weight in self.adjacency.get(node, ()):
                    alt = cost + weight
                    if alt <= max_distance_m and alt < dist.get(nxt, math.inf):
                        dist[nxt] = alt
                        heapq.heappush(queue, (alt, nxt))
            self._dist_cache[key] = dist
        return self._dist_cache[key]

    def distance_between_arcs(self, previous: int, current: int,
                              max_distance_m: float) -> float:
        """Approximate midpoint-to-midpoint distance along directed roads."""
        if previous not in self.arc or current not in self.arc:
            return math.inf
        _, pv = self.arc[previous]
        cu, _ = self.arc[current]
        plen, clen = self.edge_lengths[previous], self.edge_lengths[current]
        if previous == current:
            return 0.5 * plen
        connector = self.shortest_from(pv, max_distance_m).get(cu, math.inf)
        total = 0.5 * plen + connector + 0.5 * clen
        return total if total <= max_distance_m else math.inf

class TransitionModel:
    """Log transition score based on a reachable path and GPS displacement."""
    def __init__(self, network: DirectedRoadNetwork, *, distance_scale_m: float = 100.0,
                 max_network_distance_m: float = 5000.0, max_speed_mps: float = 70.0,
                 unreachable_log_probability: float = -30.0):
        if distance_scale_m <= 0 or max_network_distance_m <= 0 or max_speed_mps <= 0:
            raise ValueError("transition scales and limits must be positive")
        self.network = network
        self.distance_scale_m = float(distance_scale_m)
        self.max_network_distance_m = float(max_network_distance_m)
        self.max_speed_mps = float(max_speed_mps)
        self.unreachable_log_probability = float(unreachable_log_probability)

    def score(self, previous: int, current: int, gps_displacement_m: float = 0.0,
              elapsed_s: float | None = None) -> Transition:
        distance = self.network.distance_between_arcs(
            int(previous), int(current), self.max_network_distance_m)
        if not math.isfinite(distance):
            return Transition(self.unreachable_log_probability, False)
        if elapsed_s is not None and elapsed_s > 0 and distance > self.max_speed_mps * elapsed_s:
            return Transition(self.unreachable_log_probability, False, distance)
        delta = abs(distance - max(0.0, float(gps_displacement_m)))
        return Transition(-delta / self.distance_scale_m, True, distance)

    def log_probabilities(self, previous: int, candidates: Sequence[int],
                          gps_displacement_m: float = 0.0,
                          elapsed_s: float | None = None) -> dict[int, float]:
        """Normalize reachable transition likelihoods over this observation's states."""
        raw = {}
        for current in candidates:
            result = self.score(previous, int(current), gps_displacement_m, elapsed_s)
            if result.reachable:
                raw[int(current)] = result.log_probability
        if not raw:
            return {}
        peak = max(raw.values())
        log_z = peak + math.log(sum(math.exp(value - peak) for value in raw.values()))
        return {arc_id: value - log_z for arc_id, value in raw.items()}

def _safe_log_probability(value: object, floor: float) -> float:
    try:
        p = float(value)
    except (TypeError, ValueError):
        p = floor
    if not math.isfinite(p):
        p = floor
    return math.log(max(floor, min(1.0, p)))

def viterbi_decode(observations: Sequence[Sequence[Mapping]], transition_model: TransitionModel,
                   *, probability_floor: float = 1e-12) -> list[int | None]:
    """Decode ordered observations of candidate mappings {arc_id, probability}.

    Empty observations split a trip. If a layer has candidates but none are
    reachable from the prior layer, it starts a new segment from emissions.
    """
    if probability_floor <= 0 or probability_floor >= 1:
        raise ValueError("probability_floor must be in (0, 1)")
    if not observations:
        return []
    layers = []
    for obs in observations:
        layer = []
        for c in obs:
            try:
                aid = int(c["arc_id"])
            except (KeyError, TypeError, ValueError):
                continue
            layer.append({**c, "arc_id": aid,
                          "_emit": _safe_log_probability(c.get("probability"), probability_floor)})
        layers.append(layer)
    scores, back = [], []
    for t, layer in enumerate(layers):
        if not layer:
            scores.append({})
            back.append({})
            continue
        emission = {c["arc_id"]: c["_emit"] for c in layer}
        if t == 0 or not scores[t - 1]:
            scores.append(emission)
            back.append({aid: None for aid in emission})
            continue
        disp = float(layer[0].get("gps_displacement_m", 0.0) or 0.0)
        elapsed = layer[0].get("elapsed_s")
        next_scores, pointers = {}, {}
        candidate_ids = [c["arc_id"] for c in layer]
        transition_logs = {
            pc["arc_id"]: transition_model.log_probabilities(
                pc["arc_id"], candidate_ids, disp, elapsed)
            for pc in layers[t - 1] if pc["arc_id"] in scores[t - 1]
        }
        for c in layer:
            aid, opts = c["arc_id"], []
            for paid, log_probs in transition_logs.items():
                if aid in log_probs:
                    opts.append((scores[t - 1][paid] + log_probs[aid], paid))
            if opts:
                best, parent = max(opts, key=lambda x: x[0])
                next_scores[aid], pointers[aid] = best + emission[aid], parent
        if not next_scores:
            next_scores = emission
            pointers = {aid: None for aid in emission}
        scores.append(next_scores)
        back.append(pointers)
    result: list[int | None] = [None] * len(layers)
    end = len(layers) - 1
    while end >= 0:
        if not scores[end]:
            end -= 1
            continue
        start = end
        while start > 0 and back[start] and any(p is not None for p in back[start].values()):
            start -= 1
        state = max(scores[end], key=scores[end].get)
        for t in range(end, start - 1, -1):
            result[t] = state
            parent = back[t].get(state)
            if parent is None:
                break
            state = parent
        end = start - 1
    return result

def collapse_consecutive(arcs: Iterable[int | None]) -> list[int]:
    """Collapse repeated neighboring arc IDs; None observations are skipped."""
    result = []
    for aid in arcs:
        if aid is not None and (not result or int(aid) != result[-1]):
            result.append(int(aid))
    return result

def route_metrics(predicted: Sequence[int | None], reference: Sequence[int],
                  missing_predictions: int = 0) -> dict[str, float]:
    """Route metrics with explicit false-negative accounting for absent observations.

    Precision's standard denominator contains predicted arcs only, so an absent
    prediction cannot mathematically lower precision. Missing observations are
    added as false negatives to recall/F1 and as edit penalties; the separate
    miss rate and full-observation accuracy also expose them directly.
    """
    from collections import Counter
    p = [int(a) for a in predicted if a is not None]
    r = list(map(int, reference))
    missing_predictions = max(0, int(missing_predictions))
    overlap = sum((Counter(p) & Counter(r)).values())
    precision = overlap / len(p) if p else (1.0 if not r else 0.0)
    recall_denominator = len(r) + missing_predictions
    recall = overlap / recall_denominator if recall_denominator else (1.0 if not p else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    prev = list(range(len(r) + 1))
    for i, a in enumerate(p, 1):
        cur = [i]
        for j, b in enumerate(r, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (a != b)))
        prev = cur
    edit, denom = prev[-1] + missing_predictions, max(len(p) + missing_predictions, len(r))
    return {"arc_precision": precision, "arc_recall": recall, "arc_f1": f1,
            "edit_distance": float(edit),
            "normalized_sequence_similarity": 1.0 - edit / denom if denom else 1.0,
            "exact_route_match": float(p == r and missing_predictions == 0),
            "missing_prediction_count": float(missing_predictions)}
