from __future__ import annotations

import math

import pytest

from benchmark.benchmark import (
    _interval_union_duration,
    _request_meets_slo,
    _slo_capacity_round_passes,
    parse_workload_mix,
)


def test_parse_workload_mix_normalizes_weights() -> None:
    workload = parse_workload_mix("128:1024:3,4096:32:1")

    assert workload == [
        {"input": 128, "output": 1024, "weight": 0.75},
        {"input": 4096, "output": 32, "weight": 0.25},
    ]


@pytest.mark.parametrize("value", ["", "128:32", "0:32:1", "128:0:1", "128:32:0"])
def test_parse_workload_mix_rejects_invalid_entries(value: str) -> None:
    with pytest.raises(ValueError):
        parse_workload_mix(value)


def test_interval_union_duration_merges_overlapping_and_adjacent_ranges() -> None:
    duration = _interval_union_duration(
        [(0.0, 2.0), (1.0, 4.0), (6.0, 8.0), (8.0, 9.0), (5.0, 5.0), (math.nan, 2.0)]
    )

    assert duration == 7.0


def test_interval_union_duration_returns_none_without_valid_intervals() -> None:
    assert _interval_union_duration([(1.0, 1.0), (4.0, 3.0), (math.inf, 5.0)]) is None


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        ({"ttft": 1.0, "output_tokens": 2, "tpot": 0.05}, True),
        ({"ttft": 1.0, "output_tokens": 2, "tpot": 0.15}, False),
        ({"ttft": 1.5, "output_tokens": 1, "tpot": None}, True),
        ({"ttft": 2.0, "output_tokens": 2, "tpot": 0.01}, False),
        ({"ttft": None, "output_tokens": 2, "tpot": 0.01}, False),
        ({"ttft": 1.0, "output_tokens": None, "tpot": 0.01}, False),
    ],
)
def test_request_slo_evaluation(result: dict[str, float | None], expected: bool) -> None:
    assert _request_meets_slo(result, slo_ttft=1.5, slo_tpot=0.1) is expected


@pytest.mark.parametrize(
    ("metrics", "expected"),
    [
        ({"total_requests": 4, "goodput_pct": 95.0, "failure_rate": 0.0}, True),
        ({"total_requests": 4, "goodput_pct": 89.9, "failure_rate": 0.0}, False),
        ({"total_requests": 4, "goodput_pct": 95.0, "failure_rate": 0.3}, False),
        ({"total_requests": 0, "goodput_pct": 100.0, "failure_rate": 0.0}, False),
        ({"total_requests": 4, "goodput_pct": math.nan, "failure_rate": 0.0}, False),
    ],
)
def test_slo_capacity_round_evaluation(metrics: dict[str, object], expected: bool) -> None:
    assert _slo_capacity_round_passes(metrics, max_failure_rate=0.2, min_goodput_pct=90.0) is expected
