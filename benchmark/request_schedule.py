"""生成可复现的开放式请求到达计划。"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random


@dataclass
class ArrivalPlan:
    """描述请求的相对到达时间和对应速率。"""

    due_times_seconds: list[float]
    request_rates: list[float | None]
    schedule: str
    burstiness: float | None
    ramp_up_strategy: str


def build_arrival_plan(
    request_count: int,
    *,
    target_rps: float | None,
    rate_schedule: str = "fixed",
    rate_burstiness: float | None = None,
    rate_seed: int | None = None,
    rate_ramp_up_strategy: str = "none",
    rate_ramp_up_start_rps: float | None = None,
    rate_ramp_up_end_rps: float | None = None,
) -> ArrivalPlan:
    """构建 vLLM 风格的开放式请求到达计划。

    无速率限制时，所有请求均在相对零秒到达。固定、泊松和 gamma
    调度分别生成固定间隔、shape 为 1 的 gamma 间隔和指定 shape 的
    gamma 间隔。未启用 ramp 时，随机调度会归一化为目标总时长。

    Args:
        request_count: 计划中的非负请求数。
        target_rps: 未启用 ramp 时的目标请求速率；None 表示不限制速率。
        rate_schedule: 到达间隔分布，可为 fixed、poisson 或 gamma。
        rate_burstiness: gamma 调度的正 shape 参数；无穷大表示固定间隔。
        rate_seed: 随机调度使用的可选随机种子。
        rate_ramp_up_strategy: 速率爬升策略，可为 none、linear 或 exponential。
        rate_ramp_up_start_rps: ramp 起始请求速率。
        rate_ramp_up_end_rps: ramp 结束请求速率。

    Returns:
        每个请求的相对到达时间和生成该间隔时采用的速率。

    Raises:
        ValueError: 当请求数、调度参数或 ramp 参数无效时抛出。
    """
    _validate_request_count(request_count)
    _validate_schedule(rate_schedule, rate_burstiness)
    _validate_seed(rate_seed)

    if rate_ramp_up_strategy not in {"none", "linear", "exponential"}:
        raise ValueError(
            "rate_ramp_up_strategy must be 'none', 'linear', or 'exponential'"
        )

    if rate_ramp_up_strategy == "none":
        if (
            rate_ramp_up_start_rps is not None
            or rate_ramp_up_end_rps is not None
        ):
            raise ValueError(
                "rate_ramp_up_start_rps and rate_ramp_up_end_rps require "
                "rate_ramp_up_strategy"
            )
        if target_rps is None:
            return ArrivalPlan(
                due_times_seconds=[0.0] * request_count,
                request_rates=[None] * request_count,
                schedule="unbounded",
                burstiness=None,
                ramp_up_strategy="none",
            )
        _validate_positive_finite(target_rps, "target_rps")
        rates = [float(target_rps)] * request_count
        due_times = _build_due_times(rates, rate_schedule, rate_burstiness, rate_seed)
        if rate_schedule != "fixed" and due_times:
            _normalize_due_times(due_times, request_count / float(target_rps))
    else:
        if target_rps is not None:
            raise ValueError(
                "target_rps must be None when rate_ramp_up_strategy is enabled"
            )
        _validate_positive_finite(
            rate_ramp_up_start_rps, "rate_ramp_up_start_rps"
        )
        _validate_positive_finite(rate_ramp_up_end_rps, "rate_ramp_up_end_rps")
        rates = _build_ramp_rates(
            request_count,
            float(rate_ramp_up_start_rps),
            float(rate_ramp_up_end_rps),
            rate_ramp_up_strategy,
        )
        due_times = _build_due_times(rates, rate_schedule, rate_burstiness, rate_seed)

    return ArrivalPlan(
        due_times_seconds=due_times,
        request_rates=rates,
        schedule=rate_schedule,
        burstiness=rate_burstiness if rate_schedule == "gamma" else None,
        ramp_up_strategy=rate_ramp_up_strategy,
    )


def _build_due_times(
    rates: list[float],
    rate_schedule: str,
    rate_burstiness: float | None,
    rate_seed: int | None,
) -> list[float]:
    """按给定速率累积生成每条请求的到达时间。"""
    random_generator = random.Random(rate_seed)
    due_times: list[float] = []
    due_at = 0.0
    for rate in rates:
        interval = _next_interval(
            rate, rate_schedule, rate_burstiness, random_generator
        )
        due_at += interval
        due_times.append(due_at)
    return due_times


def _build_ramp_rates(
    request_count: int,
    start_rps: float,
    end_rps: float,
    strategy: str,
) -> list[float]:
    """按请求索引插值得到 ramp 期间的速率。"""
    denominator = max(request_count - 1, 1)
    rates = []
    for index in range(request_count):
        progress = index / denominator
        if strategy == "linear":
            rate = start_rps + (end_rps - start_rps) * progress
        else:
            rate = start_rps * (end_rps / start_rps) ** progress
        rates.append(rate)
    return rates


def _next_interval(
    rate: float,
    rate_schedule: str,
    rate_burstiness: float | None,
    random_generator: random.Random,
) -> float:
    """生成一个均值为 1/rate 的请求到达间隔。"""
    if rate_schedule == "fixed":
        return 1.0 / rate
    shape = 1.0 if rate_schedule == "poisson" else rate_burstiness
    assert shape is not None
    if math.isinf(shape):
        return 1.0 / rate
    return random_generator.gammavariate(shape, 1.0 / (shape * rate))


def _normalize_due_times(due_times: list[float], target_duration: float) -> None:
    """就地缩放累计到达时间，使末尾匹配目标总时长。"""
    raw_duration = due_times[-1]
    if raw_duration <= 0.0:
        raise ValueError("random arrival intervals must have a positive total duration")
    scale = target_duration / raw_duration
    for index, due_time in enumerate(due_times):
        due_times[index] = due_time * scale


def _validate_request_count(request_count: int) -> None:
    """验证请求数是非负整数。"""
    if isinstance(request_count, bool) or not isinstance(request_count, int):
        raise ValueError("request_count must be a non-negative integer")
    if request_count < 0:
        raise ValueError("request_count must be a non-negative integer")


def _validate_schedule(rate_schedule: str, rate_burstiness: float | None) -> None:
    """验证调度类型及其 burstiness 参数。"""
    if rate_schedule not in {"fixed", "poisson", "gamma"}:
        raise ValueError("rate_schedule must be 'fixed', 'poisson', or 'gamma'")
    if rate_schedule == "gamma":
        _validate_positive_burstiness(rate_burstiness, "rate_burstiness")
    elif rate_burstiness is not None:
        raise ValueError("rate_burstiness must be None unless rate_schedule is 'gamma'")


def _validate_seed(rate_seed: int | None) -> None:
    """验证可选随机种子。"""
    if rate_seed is not None and (
        isinstance(rate_seed, bool) or not isinstance(rate_seed, int)
    ):
        raise ValueError("rate_seed must be an integer or None")


def _validate_positive_burstiness(value: float | None, name: str) -> None:
    """验证 burstiness 是正数或表示固定间隔的正无穷。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive number or infinity")
    if math.isnan(value) or value <= 0:
        raise ValueError(f"{name} must be a positive number or infinity")


def _validate_positive_finite(value: float | None, name: str) -> None:
    """验证值是正且有限的数值。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive finite number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
