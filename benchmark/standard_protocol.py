"""Versioned standard benchmark protocol definitions and report summaries."""

from __future__ import annotations

import hashlib
import json
from typing import Any

STANDARD_V1_PROTOCOL_ID = "standard-v1"
STANDARD_V1_WORKLOAD_IDS = (
    "latency-short",
    "prefill-long-context",
    "decode-long-output",
    "prefix-cache",
    "concurrency-capacity",
    "mixed-production",
)
STANDARD_SUMMARY_SCHEMA_VERSION = 1

_STANDARD_METRIC_KEYS = (
    "concurrency",
    "total_requests",
    "successful",
    "failed",
    "failure_rate",
    "goodput_pct",
    "qps",
    "overall_throughput",
    "prefill_throughput",
    "decode_throughput",
    "avg_ttft",
    "p50_ttft",
    "p90_ttft",
    "p99_ttft",
    "avg_tpot",
    "p50_tpot",
    "p90_tpot",
    "p99_tpot",
)
_STANDARD_WORKLOAD_PARAM_KEYS = (
    "mode",
    "api_transport",
    "dataset",
    "context_len",
    "max_tokens",
    "random_input_len",
    "random_output_len",
    "random_prefix_len",
    "random_range_ratio",
    "random_seed",
    "concurrency",
    "num_prompts",
    "share_prefix",
    "prefix_ratio",
    "ignore_eos",
    "no_warmup",
    "warmup_rounds",
    "warmup_requests_per_round",
    "slo_ttft",
    "slo_tpot",
    "max_failure_rate",
    "min_goodput_pct",
    "sweep_max_concurrency",
    "sweep_requests_per_round",
    "slo_capacity_search_strategy",
    "slo_capacity_linear_step",
    "slo_capacity_confirm_window",
    "slo_capacity_confirm_rounds",
    "workload_mix",
)


class StandardProtocolError(ValueError):
    """Raised when a suite does not satisfy a standard protocol contract."""


def get_protocol_id(config: dict[str, Any]) -> str | None:
    """Return the configured protocol ID after structural validation.

    Args:
        config: Expanded suite configuration root.

    Returns:
        The configured protocol ID, or None for a non-standard suite.

    Raises:
        StandardProtocolError: If the protocol declaration is malformed.
    """
    protocol = config.get("protocol")
    if protocol is None:
        return None
    if not isinstance(protocol, dict):
        raise StandardProtocolError("protocol must be an object")
    unknown = sorted(set(protocol) - {"id"})
    if unknown:
        raise StandardProtocolError(
            f"protocol has unknown fields: {', '.join(unknown)}"
        )
    protocol_id = protocol.get("id")
    if not isinstance(protocol_id, str) or not protocol_id:
        raise StandardProtocolError("protocol.id must be a non-empty string")
    if protocol_id != STANDARD_V1_PROTOCOL_ID:
        raise StandardProtocolError(
            f"protocol.id must be {STANDARD_V1_PROTOCOL_ID!r}"
        )
    return protocol_id


def validate_protocol(config: dict[str, Any]) -> str | None:
    """Validate a suite's optional standard benchmark protocol contract.

    Args:
        config: Structurally valid suite configuration root.

    Returns:
        The protocol ID, or None when the suite has no standard protocol.

    Raises:
        StandardProtocolError: If standard workload declarations are incomplete
            or violate the standard-v1 contract.
    """
    protocol_id = get_protocol_id(config)
    cases = config.get("cases", [])
    has_standard_workloads = any(
        isinstance(case, dict) and "standard_workload" in case for case in cases
    )
    if protocol_id is None:
        if has_standard_workloads:
            raise StandardProtocolError(
                "standard_workload requires protocol.id='standard-v1'"
            )
        return None

    workloads: list[str] = []
    for index, case in enumerate(cases):
        location = f"cases[{index}]"
        workload = case.get("standard_workload")
        if workload not in STANDARD_V1_WORKLOAD_IDS:
            allowed = ", ".join(STANDARD_V1_WORKLOAD_IDS)
            raise StandardProtocolError(
                f"{location}.standard_workload must be one of: {allowed}"
            )
        if case.get("matrix"):
            raise StandardProtocolError(
                f"{location}.matrix is not allowed by {STANDARD_V1_PROTOCOL_ID}"
            )
        if case.get("repeat", 1) != 1:
            raise StandardProtocolError(
                f"{location}.repeat must be 1 for {STANDARD_V1_PROTOCOL_ID}"
            )
        if not case.get("enabled", True):
            raise StandardProtocolError(
                f"{location}.enabled must be true for {STANDARD_V1_PROTOCOL_ID}"
            )
        workloads.append(workload)

    duplicate_workloads = sorted(
        workload for workload in set(workloads) if workloads.count(workload) > 1
    )
    if duplicate_workloads:
        raise StandardProtocolError(
            "duplicate standard_workload values: "
            f"{', '.join(duplicate_workloads)}"
        )
    missing_workloads = sorted(set(STANDARD_V1_WORKLOAD_IDS) - set(workloads))
    if missing_workloads:
        raise StandardProtocolError(
            "missing standard-v1 workloads: "
            f"{', '.join(missing_workloads)}"
        )
    return protocol_id


def build_standard_summary(report: dict[str, Any]) -> dict[str, Any] | None:
    """Build a deterministic, target-neutral summary for a standard-v1 report.

    Args:
        report: In-progress or terminal suite report.

    Returns:
        A standard-v1 summary without timestamps, paths, host data, or endpoint
        identifiers; None when the report does not use a standard protocol.
    """
    suite = report.get("suite")
    if not isinstance(suite, dict) or suite.get("protocol_id") != STANDARD_V1_PROTOCOL_ID:
        return None

    records = report.get("cases")
    records_by_workload = {
        record.get("standard_workload"): record
        for record in records if isinstance(record, dict)
        and isinstance(record.get("standard_workload"), str)
    } if isinstance(records, list) else {}
    workloads = [
        _standard_workload_summary(workload_id, records_by_workload.get(workload_id))
        for workload_id in STANDARD_V1_WORKLOAD_IDS
    ]
    statuses = [workload["status"] for workload in workloads]
    if "failed" in statuses:
        outcome = "failed"
    elif statuses and all(status == "passed" for status in statuses):
        outcome = "passed"
    else:
        outcome = "inconclusive"

    contract = [
        {
            "id": workload["id"],
            "scenario": workload["scenario"],
            "signature_sha256": workload["signature_sha256"],
        }
        for workload in workloads
    ]
    return {
        "schema_version": STANDARD_SUMMARY_SCHEMA_VERSION,
        "protocol_id": STANDARD_V1_PROTOCOL_ID,
        "outcome": outcome,
        "contract_sha256": _stable_hash(contract),
        "workloads": workloads,
    }


def _standard_workload_summary(
    workload_id: str, record: dict[str, Any] | None
) -> dict[str, Any]:
    """Build one canonical workload summary from a suite case record."""
    if record is None:
        return {
            "id": workload_id,
            "status": "missing",
            "scenario": None,
            "signature_sha256": _stable_hash({"id": workload_id}),
            "metrics": {key: None for key in _STANDARD_METRIC_KEYS},
        }

    scenario = record.get("scenario")
    matrix = record.get("matrix") if isinstance(record.get("matrix"), dict) else {}
    params = record.get("params") if isinstance(record.get("params"), dict) else {}
    contract = {
        "id": workload_id,
        "scenario": scenario,
        "matrix": matrix,
        "params": {
            key: params.get(key)
            for key in _STANDARD_WORKLOAD_PARAM_KEYS
            if key in params
        },
    }
    return {
        "id": workload_id,
        "status": record.get("status", "missing"),
        "scenario": scenario,
        "signature_sha256": _stable_hash(contract),
        "metrics": _standard_metrics(record.get("result")),
    }


def _standard_metrics(result: Any) -> dict[str, Any]:
    """Return fixed-shape primary metrics from a scenario result."""
    metrics: dict[str, Any] = {}
    if isinstance(result, dict):
        selected_metrics = result.get("selected_metrics")
        if isinstance(selected_metrics, dict):
            metrics = selected_metrics
        elif isinstance(result.get("metrics"), dict):
            metrics = result["metrics"]
    return {key: metrics.get(key) for key in _STANDARD_METRIC_KEYS}


def _stable_hash(value: Any) -> str:
    """Return a stable SHA-256 digest for a JSON-compatible protocol value."""
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
