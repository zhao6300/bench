from __future__ import annotations

import math
import sys
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


def test_rich_runtime_output_is_recorded_as_dashboard_events(capsys) -> None:
    """Keep execution-time stdout and stderr out of the Rich alternate screen."""
    events: list[str] = []
    reporter = SimpleNamespace(captures_runtime_output=True, event=events.append)
    args = SimpleNamespace(_progress_reporter=reporter)

    with benchmark_module._redirect_runtime_output(args):
        print("加载 tokenizer: placeholder")
        print("预热警告: timeout", file=sys.stderr)
        print("扫描档位", end="")

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert events == ["加载 tokenizer: placeholder", "预热警告: timeout", "扫描档位"]


def test_runtime_output_keeps_plain_and_off_streams_unchanged(capsys) -> None:
    """Do not intercept execution output for non-Rich progress reporters."""
    args = SimpleNamespace(_progress_reporter=SimpleNamespace(captures_runtime_output=False))

    with benchmark_module._redirect_runtime_output(args):
        print("正常输出")
        print("错误输出", file=sys.stderr)

    captured = capsys.readouterr()
    assert captured.out == "正常输出\n"
    assert captured.err == "错误输出\n"
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


def test_pd_ratio_uses_business_shapes_and_respects_ignore_eos(monkeypatch) -> None:
    """Use actual batch lengths for P:D sizing and preserve EOS configuration."""
    import sys
    import types

    progress_events: list[tuple[object, ...]] = []

    class _ProgressSpy:
        """Capture dashboard-only callbacks without a terminal renderer."""

        def stage_started(self, stage: str, detail: str | None = None) -> None:
            progress_events.append(("stage_started", stage, detail))

        def stage_finished(self, stage: str, detail: str | None = None) -> None:
            progress_events.append(("stage_finished", stage, detail))

        def event(self, message: str) -> None:
            progress_events.append(("event", message))

    args = SimpleNamespace(
        avg_input_tokens=1000,
        avg_output_tokens=5000,
        total_gpus=None,
        tokenizer="test-tokenizer",
        model="test-model",
        api_base="http://localhost:8000/v1",
        api_key=None,
        no_warmup=True,
        dataset="random",
        ignore_eos=False,
        slo_ttft=60.0,
        slo_tpot=0.05,
        api_transport="requests",
        _progress_reporter=_ProgressSpy(),
    )
    batches = iter([
        SimpleNamespace(
            prompts=["prefill"] * 8,
            prompt_lens=[800] * 8,
            output_lens=[1] * 8,
        ),
        SimpleNamespace(
            prompts=["decode"] * 8,
            prompt_lens=[800] * 8,
            output_lens=[5000] * 8,
        ),
    ])
    workloads = iter([
        {
            "prompt_tokens": {"min": 800, "max": 800, "avg": 800.0, "total": 6400},
            "requested_output_tokens": {"min": 1, "max": 1, "avg": 1.0, "total": 8},
        },
        {
            "prompt_tokens": {"min": 800, "max": 800, "avg": 800.0, "total": 6400},
            "requested_output_tokens": {
                "min": 5000, "max": 5000, "avg": 5000.0, "total": 40000,
            },
        },
    ])
    round_metrics = iter([
        {
            "successful": 8,
            "prefill_throughput": 100.0,
            "avg_ttft": 0.1,
            "p99_ttft": 0.2,
        },
        {
            "successful": 8,
            "decode_throughput": 100.0,
            "avg_tpot": 0.01,
            "avg_estimated_itl": None,
        },
    ])
    build_calls: list[dict[str, object]] = []
    round_calls: list[dict[str, object]] = []
    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: object())
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    def build_batch(*_args: object, **kwargs: object) -> SimpleNamespace:
        build_calls.append(kwargs)
        return next(batches)

    def run_round(*round_args: object, **_kwargs: object) -> dict[str, object]:
        round_calls.append({
            "prompt_lens": round_args[1],
            "max_tokens": round_args[5],
            "ignore_eos": round_args[8],
        })
        return next(round_metrics)

    monkeypatch.setattr(benchmark_module, "build_request_batch", build_batch)
    monkeypatch.setattr(benchmark_module, "summarize_dataset_batch", lambda *_args: next(workloads))
    monkeypatch.setattr(benchmark_module, "run_api_benchmark_round", run_round)

    result = benchmark_module.run_pd_ratio_benchmark(args)

    assert build_calls == [
        {"input_len": 1000, "output_len": 1, "share_prefix": False,
         "random_range_ratio": 0.0, "run_id": build_calls[0]["run_id"]},
        {"input_len": 1000, "output_len": 5000, "share_prefix": False,
         "random_range_ratio": 0.0, "run_id": build_calls[1]["run_id"]},
    ]
    assert round_calls == [
        {"prompt_lens": [800] * 8, "max_tokens": [1] * 8, "ignore_eos": False},
        {"prompt_lens": [800] * 8, "max_tokens": [5000] * 8, "ignore_eos": False},
    ]
    assert progress_events == [
        ("stage_started", "加载 tokenizer", "test-tokenizer"),
        ("stage_finished", "加载 tokenizer", None),
        ("event", "跳过预热请求"),
        ("stage_started", "构造 Prefill 测量负载", None),
        ("stage_finished", "构造 Prefill 测量负载", None),
        ("stage_started", "测量 Prefill 吞吐量", None),
        ("stage_finished", "测量 Prefill 吞吐量", None),
        ("stage_started", "构造 Decode 测量负载", None),
        ("stage_finished", "构造 Decode 测量负载", None),
        ("stage_started", "测量 Decode 吞吐量", None),
        ("stage_finished", "测量 Decode 吞吐量", None),
        ("stage_started", "计算 P:D 容量比例", None),
        ("stage_finished", "计算 P:D 容量比例", None),
    ]
    assert result["analysis"]["prefill_time_seconds"] == pytest.approx(8.0)
    assert result["analysis"]["decode_time_seconds"] == pytest.approx(50.0)
    assert result["analysis"]["business_workload"] == {
        "configured_avg_input_tokens": 1000,
        "configured_avg_output_tokens": 5000,
        "prefill_measurement_prompt_tokens": 800.0,
        "decode_measurement_prompt_tokens": 800.0,
        "decode_measurement_requested_output_tokens": 5000.0,
    }


def test_pd_ratio_rejects_single_token_business_output() -> None:
    """Reject a P:D workload that cannot produce a Decode throughput sample."""
    with pytest.raises(
        benchmark_module.BenchmarkConfigError,
        match="avg_output_tokens must be at least 2 for pd-ratio",
    ):
        benchmark_module._build_case_args(
            {"mode": "api"},
            {
                "name": "pd-single-token-output",
                "preset": "pd-ratio",
                "params": {"pd": {"avg_output_tokens": 1}},
            },
            {},
        )
