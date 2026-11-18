"""Synthetic directed-network tests for Member 3 map matching."""
import pandas as pd
from pathsnap.hmm import (DirectedRoadNetwork, TransitionModel, collapse_consecutive,
                          route_metrics, viterbi_decode)

def network(rows, lengths=None):
    arcs = pd.DataFrame(rows, columns=["arc_id", "from_node_id", "to_node_id"])
    return DirectedRoadNetwork(arcs, lengths or {})

def test_directed_connection_and_reverse_unreachable():
    net = network([(0,0,1),(1,1,2)], {0:10,1:10})
    assert net.distance_between_arcs(0,1,100) == 10
    assert net.distance_between_arcs(1,0,100) == float("inf")

def test_multiple_intermediate_arcs():
    net = network([(0,0,1),(1,1,2),(2,2,3),(3,3,4)], {i:10 for i in range(4)})
    assert net.distance_between_arcs(0,3,100) == 30

def test_viterbi_chooses_global_path_over_local_peak():
    net = network([(0,0,1),(1,2,3),(2,3,4)], {0:10,1:10,2:10})
    tm = TransitionModel(net, distance_scale_m=100)
    got = viterbi_decode([
        [{"arc_id":0,"probability":0.55},{"arc_id":1,"probability":0.45}],
        [{"arc_id":2,"probability":0.99}],
    ], tm)
    assert got == [1,2]

def test_known_chain_and_backtracking():
    net = network([(0,0,1),(1,1,2),(2,2,3)], {0:10,1:10,2:10})
    tm = TransitionModel(net)
    got = viterbi_decode([
        [{"arc_id":0,"probability":0.9},{"arc_id":2,"probability":0.1}],
        [{"arc_id":1,"probability":0.6},{"arc_id":2,"probability":0.4}],
        [{"arc_id":2,"probability":0.9}],
    ], tm)
    assert got == [0,1,2]

def test_disconnected_transition_restarts_at_local_emission():
    net = network([(0,0,1),(1,5,6)])
    got = viterbi_decode([[{"arc_id":0,"probability":0.9}],
                          [{"arc_id":1,"probability":0.8}]], TransitionModel(net))
    assert got == [0,1]

def test_no_candidate_observation_splits_sequence():
    net = network([(0,0,1)])
    got = viterbi_decode([[{"arc_id":0,"probability":0.9}], [],
                          [{"arc_id":0,"probability":0.8}]], TransitionModel(net))
    assert got == [0,None,0]

def test_one_observation_and_empty_input():
    tm = TransitionModel(network([(4,0,1)]))
    assert viterbi_decode([], tm) == []
    assert viterbi_decode([[{"arc_id":4,"probability":0.01}]], tm) == [4]

def test_small_missing_and_zero_probabilities_are_stable():
    tm = TransitionModel(network([(1,0,1)]))
    got = viterbi_decode([[{"arc_id":1,"probability":0.0}],
                          [{"arc_id":1,"probability":None}]], tm)
    assert got == [1,1]

def test_repeated_arcs_collapse_and_metrics_match_reference():
    assert collapse_consecutive([2,2,3,3,2]) == [2,3,2]
    m = route_metrics([1,2,3], [1,2,4])
    assert m["arc_precision"] == 2/3
    assert m["arc_recall"] == 2/3
    assert m["arc_f1"] == 2/3
    assert m["edit_distance"] == 1
    assert abs(m["normalized_sequence_similarity"] - 2/3) < 1e-12
    assert route_metrics([], [])["exact_route_match"] == 1.0


def test_transition_probabilities_normalize_over_reachable_candidates():
    net = network([(0,0,1),(1,1,2),(2,1,3)], {0:10,1:10,2:20})
    tm = TransitionModel(net)
    logs = tm.log_probabilities(0, [1,2])
    assert set(logs) == {1,2}
    assert abs(sum(__import__("math").exp(v) for v in logs.values()) - 1.0) < 1e-12


def test_missing_observations_are_counted_as_route_false_negatives():
    result = route_metrics([1], [1], missing_predictions=1)
    assert result["arc_precision"] == 1.0
    assert result["arc_recall"] == 0.5
    assert result["missing_prediction_count"] == 1
    assert result["edit_distance"] == 1.0
    assert result["exact_route_match"] == 0.0
