from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark import benchmark as benchmark_module
from benchmark.benchmark import BenchmarkConfigError, build_parser, load_suite_config
from benchmark.standard_protocol import (
    STANDARD_V1_PROTOCOL_ID,
    STANDARD_V1_WORKLOAD_IDS,
    build_standard_summary,
)


_STANDARD_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "examples" / "benchmark-config-standard-v1.json"
)


def _metrics() -> dict[str, object]:
    """Return a complete successful metric sample for protocol tests."""
    return {
        "concurrency": 4,
        "total_requests": 8,
        "successful": 8,
        "failed": 0,
        "failure_rate": 0.0,
        "goodput_pct": 100.0,
        "qps": 2.0,
        "overall_throughput": 256.0,
        "prefill_throughput": 512.0,
        "decode_throughput": 128.0,
        "avg_ttft": 0.1,
        "p50_ttft": 0.09,
        "p90_ttft": 0.15,
        "p99_ttft": 0.2,
        "avg_tpot": 0.01,
        "p50_tpot": 0.009,
        "p90_tpot": 0.015,
        "p99_tpot": 0.02,
    }


def _standard_report() -> dict[str, object]:
    """Build a successful report with unstable fields omitted from the contract."""
    return {
        "suite": {"protocol_id": STANDARD_V1_PROTOCOL_ID},
        "cases": [
            {
                "standard_workload": workload_id,
                "status": "passed",
                "scenario": "single",
                "matrix": {},
                "params": {
                    "api_base": "http://localhost:8000/v1",
                    "model": "different-per-target",
                    "random_input_len": 512,
                    "random_output_len": 128,
                    "concurrency": 4,
                },
                "result": {"scenario": "single", "metrics": _metrics()},
            }
            for workload_id in STANDARD_V1_WORKLOAD_IDS
        ],
    }


def test_standard_v1_example_is_a_complete_valid_protocol() -> None:
    """Keep the published standard-v1 configuration aligned with its contract."""
    config = load_suite_config(str(_STANDARD_CONFIG_PATH))

    assert config["protocol"] == {"id": STANDARD_V1_PROTOCOL_ID}
    assert [case["standard_workload"] for case in config["cases"]] == list(
        STANDARD_V1_WORKLOAD_IDS
    )


def test_standard_v1_rejects_missing_required_workload(tmp_path) -> None:
    """Reject partial standards that could not produce a comparable report."""
    config_path = tmp_path / "partial-standard.json"
    config_path.write_text(json.dumps({
        "version": 1,
        "protocol": {"id": STANDARD_V1_PROTOCOL_ID},
        "cases": [
            {"name": workload_id, "standard_workload": workload_id}
            for workload_id in STANDARD_V1_WORKLOAD_IDS[:-1]
        ],
    }), encoding="utf-8")

    with pytest.raises(BenchmarkConfigError, match="missing standard-v1 workloads"):
        load_suite_config(str(config_path))


def test_standard_summary_is_stable_and_excludes_target_identifiers() -> None:
    """Keep baseline-comparison fields independent of endpoint and model names."""
    report = _standard_report()
    summary = build_standard_summary(report)
    assert summary is not None

    report["cases"].reverse()
    reordered_summary = build_standard_summary(report)

    assert summary == reordered_summary
    assert summary["outcome"] == "passed"
    assert summary["workloads"][0]["id"] == "latency-short"
    assert "api_base" not in json.dumps(summary)
    assert "different-per-target" not in json.dumps(summary)


def test_standard_v1_suite_writes_a_comparable_summary_without_network(
    monkeypatch, tmp_path
) -> None:
    """Write standard_summary through the normal suite checkpoint lifecycle."""
    report_path = tmp_path / "standard-report.json"

    def execute(_args, scenario: str) -> dict[str, object]:
        if scenario == "slo-capacity-search":
            return {"scenario": scenario, "selected_metrics": _metrics()}
        return {"scenario": scenario, "metrics": _metrics()}

    monkeypatch.setattr(benchmark_module, "execute_benchmark", execute)
    cli_args = build_parser().parse_args([
        "--config", str(_STANDARD_CONFIG_PATH), "--report", str(report_path), "--no-resume",
    ])

    assert benchmark_module.run_configured_suite(str(_STANDARD_CONFIG_PATH), cli_args) == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    summary = report["standard_summary"]

    assert summary["protocol_id"] == STANDARD_V1_PROTOCOL_ID
    assert summary["outcome"] == "passed"
    assert [workload["id"] for workload in summary["workloads"]] == list(
        STANDARD_V1_WORKLOAD_IDS
    )


def test_standard_summary_contract_changes_with_measurement_parameters() -> None:
    """Prevent comparisons when a workload changes its measurement contract."""
    report = _standard_report()
    summary = build_standard_summary(report)
    assert summary is not None

    report["cases"][0]["params"]["warmup_rounds"] = 2
    changed_summary = build_standard_summary(report)
    assert changed_summary is not None

    assert changed_summary["contract_sha256"] != summary["contract_sha256"]
