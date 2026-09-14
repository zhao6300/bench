from __future__ import annotations

import math
import random

import pytest

from benchmark.request_schedule import build_arrival_plan


def test_build_arrival_plan_returns_unbounded_zero_deadlines() -> None:
    """不限制速率时，全部请求应立即到达。"""
    plan = build_arrival_plan(3, target_rps=None)

    assert plan.due_times_seconds == [0.0, 0.0, 0.0]
    assert plan.request_rates == [None, None, None]
    assert plan.schedule == "unbounded"
    assert plan.burstiness is None
    assert plan.ramp_up_strategy == "none"


def test_build_arrival_plan_builds_fixed_intervals() -> None:
    """固定速率应产生等间隔且从第一个间隔开始的到达时间。"""
    plan = build_arrival_plan(3, target_rps=4.0)

    assert plan.due_times_seconds == pytest.approx([0.25, 0.5, 0.75])
    assert plan.request_rates == [4.0, 4.0, 4.0]
    assert plan.schedule == "fixed"
    assert plan.burstiness is None


def test_build_arrival_plan_makes_poisson_schedule_seed_reproducible() -> None:
    """相同随机种子必须生成相同的泊松到达计划。"""
    first = build_arrival_plan(
        4, target_rps=8.0, rate_schedule="poisson", rate_seed=17
    )
    second = build_arrival_plan(
        4, target_rps=8.0, rate_schedule="poisson", rate_seed=17
    )

    assert first == second


def test_build_arrival_plan_uses_seeded_gamma_intervals() -> None:
    """gamma 调度应使用给定 shape 和种子生成可预测间隔。"""
    rate = 10.0
    shape = 0.5
    random_generator = random.Random(29)
    intervals = [
        random_generator.gammavariate(shape, 1.0 / (shape * rate))
        for _ in range(3)
    ]
    raw_due_times = [
        intervals[0],
        intervals[0] + intervals[1],
        intervals[0] + intervals[1] + intervals[2],
    ]
    expected_due_times = [
        due_time * (3.0 / rate) / raw_due_times[-1]
        for due_time in raw_due_times
    ]

    plan = build_arrival_plan(
        3,
        target_rps=rate,
        rate_schedule="gamma",
        rate_burstiness=shape,
        rate_seed=29,
    )

    assert plan.due_times_seconds == pytest.approx(expected_due_times)
    assert plan.burstiness == shape


@pytest.mark.parametrize(
    "rate_schedule, rate_burstiness",
    [("poisson", None), ("gamma", 2.0)],
)
def test_build_arrival_plan_normalizes_non_ramp_random_duration(
    rate_schedule: str,
    rate_burstiness: float | None,
) -> None:
    """未启用 ramp 的随机调度末尾应严格匹配目标总时长。"""
    plan = build_arrival_plan(
        5,
        target_rps=4.0,
        rate_schedule=rate_schedule,
        rate_burstiness=rate_burstiness,
        rate_seed=5,
    )

    assert plan.due_times_seconds[-1] == pytest.approx(5.0 / 4.0)


@pytest.mark.parametrize(
    ("strategy", "expected_rates"),
    [
        ("linear", [2.0, 4.0, 6.0, 8.0]),
        ("exponential", [2.0, 4.0, 8.0, 16.0]),
    ],
)
def test_build_arrival_plan_interpolates_ramp_rate_endpoints(
    strategy: str,
    expected_rates: list[float],
) -> None:
    """线性和指数 ramp 都应包含指定的速率端点。"""
    plan = build_arrival_plan(
        4,
        target_rps=None,
        rate_ramp_up_strategy=strategy,
        rate_ramp_up_start_rps=2.0,
        rate_ramp_up_end_rps=expected_rates[-1],
    )

    assert plan.request_rates == pytest.approx(expected_rates)
    expected_due_times = [
        math.fsum(1.0 / rate for rate in expected_rates[: index + 1])
        for index in range(4)
    ]
    assert plan.due_times_seconds == pytest.approx(expected_due_times)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"target_rps": 0.0}, "target_rps"),
        ({"target_rps": float("nan")}, "target_rps"),
        ({"target_rps": True}, "target_rps"),
        ({"target_rps": 1.0, "rate_schedule": "unsupported"}, "rate_schedule"),
        ({"target_rps": 1.0, "rate_burstiness": 2.0}, "rate_burstiness"),
        ({"target_rps": 1.0, "rate_schedule": "gamma"}, "rate_burstiness"),
        (
            {
                "target_rps": 1.0,
                "rate_schedule": "gamma",
                "rate_burstiness": float("nan"),
            },
            "rate_burstiness",
        ),
        (
            {
                "target_rps": 1.0,
                "rate_ramp_up_strategy": "linear",
                "rate_ramp_up_start_rps": 1.0,
                "rate_ramp_up_end_rps": 2.0,
            },
            "target_rps",
        ),
        (
            {
                "target_rps": None,
                "rate_ramp_up_strategy": "exponential",
                "rate_ramp_up_start_rps": 0.0,
                "rate_ramp_up_end_rps": 2.0,
            },
            "rate_ramp_up_start_rps",
        ),
        (
            {
                "target_rps": None,
                "rate_ramp_up_strategy": "linear",
                "rate_ramp_up_start_rps": 1.0,
            },
            "rate_ramp_up_end_rps",
        ),
        ({"target_rps": 1.0, "rate_ramp_up_strategy": "unsupported"}, "rate_ramp_up_strategy"),
    ],
)
def test_build_arrival_plan_rejects_invalid_combinations_and_values(
    kwargs: dict[str, object],
    message: str,
) -> None:
    """无效的调度组合和数值必须产生明确错误。"""
    with pytest.raises(ValueError, match=message):
        build_arrival_plan(2, **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("request_count", [-1, True, 1.5])
def test_build_arrival_plan_rejects_invalid_request_count(
    request_count: float,
) -> None:
    """请求数仅接受非负整数。"""
    with pytest.raises(ValueError, match="request_count"):
        build_arrival_plan(request_count, target_rps=1.0)  # type: ignore[arg-type]


def test_build_arrival_plan_treats_infinite_gamma_burstiness_as_fixed() -> None:
    """无穷 burstiness 应退化为固定到达间隔。"""
    plan = build_arrival_plan(
        3,
        target_rps=4.0,
        rate_schedule="gamma",
        rate_burstiness=float("inf"),
        rate_seed=17,
    )

    assert plan.due_times_seconds == pytest.approx([0.25, 0.5, 0.75])
    assert plan.burstiness is not None
    assert math.isinf(plan.burstiness)
