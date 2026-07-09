from __future__ import annotations

import math

import networkx as nx
import pytest
import requests

from src.utils import (
    calculate_metrics,
    calculate_weighted_metrics,
    get_osrm_distance,
    haversine_distance,
    weighted_mean,
)


class FakeResponse:
    def __init__(self, payload, status_error=None):
        self.payload = payload
        self.status_error = status_error

    def raise_for_status(self):
        if self.status_error:
            raise self.status_error

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def get(self, url, timeout):
        self.calls.append((url, timeout))
        return self.response


def test_haversine_distance_is_zero_for_same_point():
    assert haversine_distance(52.52, 13.405, 52.52, 13.405) == 0.0


def test_haversine_distance_matches_known_city_scale():
    distance = haversine_distance(52.52, 13.405, 48.137, 11.575)
    assert distance == pytest.approx(504, rel=0.02)


def test_distant_pair_skips_osrm(tmp_path):
    session = FakeSession(FakeResponse({"routes": [{"distance": 1.0}]}))
    result = get_osrm_distance(
        52.52,
        13.405,
        48.137,
        11.575,
        cache_path=tmp_path / "distances.db",
        session=session,
    )
    assert result == pytest.approx(504_000, rel=0.02)
    assert session.calls == []


def test_nearby_pair_uses_osrm_and_cache(tmp_path):
    session = FakeSession(FakeResponse({"routes": [{"distance": 12_345.0}]}))
    cache = tmp_path / "distances.db"

    first = get_osrm_distance(
        52.52,
        13.405,
        52.50,
        13.30,
        cache_path=cache,
        session=session,
    )
    second = get_osrm_distance(
        52.52,
        13.405,
        52.50,
        13.30,
        cache_path=cache,
        session=session,
    )

    assert first == second == 12_345.0
    assert len(session.calls) == 1
    assert session.calls[0][0].startswith("https://router.project-osrm.org/")


def test_osrm_failure_returns_none(tmp_path):
    error = requests.HTTPError("service unavailable")
    session = FakeSession(FakeResponse({}, status_error=error))
    assert (
        get_osrm_distance(
            52.52,
            13.405,
            52.50,
            13.30,
            cache_path=tmp_path / "distances.db",
            session=session,
        )
        is None
    )


def test_metrics_are_deterministic_for_connected_graph():
    graph = nx.Graph()
    graph.add_edge(1, 2, weight=10_000)
    graph.add_edge(2, 3, weight=20_000)
    first = calculate_metrics(graph, seed=7)
    second = calculate_metrics(graph, seed=7)
    assert first == second
    assert first["average_distance"] == 15.0
    assert first["diameter"] == 2


def test_isolate_only_graph_returns_defined_weighted_metrics():
    graph = nx.Graph()
    graph.add_nodes_from([1, 2])
    metrics = calculate_weighted_metrics(graph, year=2022)
    assert metrics["total_nodes"] == 2
    assert metrics["isolated_nodes"] == 2
    assert metrics["subnetwork_sizes"] == []
    assert all(
        math.isfinite(metrics[name])
        for name in ("density", "average_distance", "diameter", "average_clustering")
    )


def test_weighted_mean_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        weighted_mean([1, 2], [1])
    with pytest.raises(ValueError):
        weighted_mean([1, 2], [0, 0])
