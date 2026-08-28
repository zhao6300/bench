from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from benchmark import benchmark as benchmark_module
from benchmark.benchmark import (
    _add_vllm_prefix_cache_hit_rate,
    _interval_union_duration,
    _peak_server_metrics,
    _request_meets_slo,
    _slo_capacity_round_passes,
    _vllm_prefix_cache_counter_delta,
    parse_workload_mix,
    query_gpu_metrics,
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


def test_vllm_prefix_cache_counter_delta_aggregates_matched_label_series(
    monkeypatch,
) -> None:
    snapshots = iter([
        "\n".join([
            'vllm:prefix_cache_hits{model_name="model-a"} 10',
            'vllm:prefix_cache_queries{model_name="model-a"} 20',
            'vllm:prefix_cache_hits{model_name="model-b"} 5',
            'vllm:prefix_cache_queries{model_name="model-b"} 10',
        ]),
        "\n".join([
            'vllm:prefix_cache_hits{model_name="model-a"} 25',
            'vllm:prefix_cache_queries{model_name="model-a"} 40',
            'vllm:prefix_cache_hits{model_name="model-b"} 8',
            'vllm:prefix_cache_queries{model_name="model-b"} 14',
        ]),
    ])

    def get(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(status_code=200, text=next(snapshots))

    monkeypatch.setattr(benchmark_module, "requests", SimpleNamespace(get=get))
    start = query_gpu_metrics("http://localhost:8000/v1")
    end = query_gpu_metrics("http://localhost:8000/v1")
    delta = _vllm_prefix_cache_counter_delta(start, end)
    summary = _peak_server_metrics([])
    _add_vllm_prefix_cache_hit_rate(summary, start, end)

    assert delta == {
        "hit_rate": 75.0,
        "hit_delta": 18.0,
        "query_delta": 24.0,
        "paired_series_count": 2,
        "reset_series_count": 0,
        "zero_query_series_count": 0,
        "source": "vllm:prefix_cache_hits / vllm:prefix_cache_queries counter delta",
    }
    assert summary["metrics"]["cache_hit_rate"] == {
        "sample_count": 1,
        "min": 75.0,
        "avg": 75.0,
        "max": 75.0,
        "p50": 75.0,
        "p90": 75.0,
        "p95": 75.0,
        "p99": 75.0,
        "aggregation": "round_counter_delta",
        "hit_delta": 18.0,
        "query_delta": 24.0,
        "paired_series_count": 2,
        "reset_series_count": 0,
        "zero_query_series_count": 0,
        "max_source": "vllm:prefix_cache_hits / vllm:prefix_cache_queries counter delta",
    }


def test_vllm_prefix_cache_counter_delta_excludes_resets_and_zero_queries() -> None:
    start = {
        "_vllm_prefix_cache_counters": {
            "hits": {"{model_name=\"reset\"}": 10, "{model_name=\"idle\"}": 3},
            "queries": {"{model_name=\"reset\"}": 20, "{model_name=\"idle\"}": 5},
            "sources": {"hits": ["vllm:prefix_cache_hits"], "queries": ["vllm:prefix_cache_queries"]},
        }
    }
    end = {
        "_vllm_prefix_cache_counters": {
            "hits": {"{model_name=\"reset\"}": 2, "{model_name=\"idle\"}": 3},
            "queries": {"{model_name=\"reset\"}": 4, "{model_name=\"idle\"}": 5},
            "sources": {"hits": ["vllm:prefix_cache_hits"], "queries": ["vllm:prefix_cache_queries"]},
        }
    }

    assert _vllm_prefix_cache_counter_delta(start, end) == {
        "hit_rate": None,
        "hit_delta": 0.0,
        "query_delta": 0.0,
        "paired_series_count": 2,
        "reset_series_count": 1,
        "zero_query_series_count": 1,
        "source": "vllm:prefix_cache_hits / vllm:prefix_cache_queries counter delta",
    }


def test_sglang_cache_hit_rate_gauge_remains_a_periodic_metric(monkeypatch) -> None:
    response = SimpleNamespace(
        status_code=200,
        text="\n".join([
            'sglang:cache_hit_rate{engine=\"0\"} 0.4',
            'sglang:cache_hit_rate{engine=\"1\"} 0.7',
        ]),
    )
    monkeypatch.setattr(
        benchmark_module,
        "requests",
        SimpleNamespace(get=lambda *_args, **_kwargs: response),
    )

    snapshot = query_gpu_metrics("http://localhost:8000/v1")
    summary = _peak_server_metrics([snapshot])

    assert snapshot["cache_hit_rate"] == 70.0
    assert snapshot["cache_hit_rate_source"] == "sglang:cache_hit_rate"
    assert summary["metrics"]["cache_hit_rate"]["max"] == 70.0
    assert "vllm_prefix_cache_counter_delta" not in summary
