"""Best-effort, extensible host hardware inventory collection for reports."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
from typing import Callable, Protocol
import re


CommandRunner = Callable[[list[str]], subprocess.CompletedProcess[str]]
TextReader = Callable[[str], str | None]


@dataclass(frozen=True)
class AcceleratorProbeResult:
    """One vendor collector's structured, non-sensitive detection result."""

    collector: str
    vendor: str
    tool: str
    state: str
    devices: list[dict[str, object]]
    reason: str | None = None


class AcceleratorCollector(Protocol):
    """Extension point for static accelerator inventory collectors."""

    def collect(self, command_runner: CommandRunner) -> AcceleratorProbeResult:
        """Collect one vendor's static accelerator inventory."""


class NvidiaAcceleratorCollector:
    """Collect NVIDIA GPU name, driver, and installed memory through NVSMI."""

    def collect(self, command_runner: CommandRunner) -> AcceleratorProbeResult:
        """Collect NVIDIA accelerator details without GPU UUIDs or process data."""
        command = [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ]
        result = _run_optional_command(command_runner, command)
        if result is None:
            return _unavailable_probe("nvidia", "nvidia-smi")
        if result.returncode:
            return _failed_probe("nvidia", "nvidia-smi", "command_failed")
        devices = []
        for row in csv.reader(result.stdout.splitlines()):
            if len(row) != 4:
                continue
            index, name, driver_version, memory_mib = (item.strip() for item in row)
            devices.append({
                "vendor": "nvidia",
                "index": _integer_or_text(index),
                "name": name,
                "driver_version": driver_version,
                "memory_total_mib": _integer_or_text(memory_mib),
            })
        return AcceleratorProbeResult(
            collector="nvidia_smi",
            vendor="nvidia",
            tool="nvidia-smi",
            state="available",
            devices=devices,
        )


class HuaweiAcceleratorCollector:
    """Collect Huawei Ascend NPU identity and driver details through npu-smi."""

    def collect(self, command_runner: CommandRunner) -> AcceleratorProbeResult:
        """Collect stable fields from npu-smi info without serial or process data."""
        result = _run_optional_command(command_runner, ["npu-smi", "info"])
        if result is None:
            return _unavailable_probe("huawei", "npu-smi")
        if result.returncode:
            return _failed_probe("huawei", "npu-smi", "command_failed")
        driver_match = re.search(r"Driver\s+Version\s*:\s*([^\s|]+)", result.stdout)
        model_match = re.search(r"((?:Ascend|Atlas)[A-Za-z0-9 ._-]+)", result.stdout)
        device_ids = sorted({
            match.group(1)
            for match in re.finditer(r"^\|\s*(\d+)\s*\|", result.stdout, re.MULTILINE)
        })
        devices = [
            {
                "vendor": "huawei",
                "index": _integer_or_text(device_id),
                "name": model_match.group(1).strip() if model_match else None,
                "driver_version": driver_match.group(1) if driver_match else None,
            }
            for device_id in device_ids
        ]
        return AcceleratorProbeResult(
            collector="huawei_npu_smi",
            vendor="huawei",
            tool="npu-smi",
            state="available",
            devices=devices,
        )


class MuxiAcceleratorCollector:
    """Collect Muxi GPU inventory from common Muxi management CLI candidates."""

    _COMMANDS = (("mx-smi", "-L"), ("mthreads-gmi", "-L"))

    def collect(self, command_runner: CommandRunner) -> AcceleratorProbeResult:
        """Collect Muxi GPU labels while tolerating tool-version differences."""
        for command in self._COMMANDS:
            result = _run_optional_command(command_runner, list(command))
            if result is None:
                continue
            tool = command[0]
            if result.returncode:
                return _failed_probe("muxi", tool, "command_failed")
            devices = []
            for index, line in enumerate(result.stdout.splitlines()):
                label = line.strip()
                if not label:
                    continue
                device_match = re.search(r"(?:GPU|Device)\s*(\d+)", label, re.IGNORECASE)
                devices.append({
                    "vendor": "muxi",
                    "index": _integer_or_text(device_match.group(1)) if device_match else index,
                    "name": _strip_device_prefix(label),
                })
            return AcceleratorProbeResult(
                collector="muxi_smi",
                vendor="muxi",
                tool=tool,
                state="available",
                devices=devices,
            )
        return _unavailable_probe("muxi", "mx-smi|mthreads-gmi")


DEFAULT_ACCELERATOR_COLLECTORS: tuple[AcceleratorCollector, ...] = (
    NvidiaAcceleratorCollector(),
    HuaweiAcceleratorCollector(),
    MuxiAcceleratorCollector(),
)


def collect_host_inventory(
    *,
    command_runner: CommandRunner | None = None,
    text_reader: TextReader | None = None,
    system_name: str | None = None,
    accelerator_collectors: tuple[AcceleratorCollector, ...] = DEFAULT_ACCELERATOR_COLLECTORS,
) -> dict[str, object]:
    """Collect a JSON-safe host inventory without failing a benchmark run.

    Args:
        command_runner: Optional injectable fixed-argv command runner for tests.
        text_reader: Optional injectable reader for Linux sysfs and procfs files.
        system_name: Optional operating-system override for tests.
        accelerator_collectors: Registered vendor collectors, extendable without
            modifying the orchestration runner.

    Returns:
        Static host details, collector states, and sanitized collection errors.
    """
    try:
        return _collect_host_inventory(
            command_runner=command_runner,
            text_reader=text_reader,
            system_name=system_name,
            accelerator_collectors=accelerator_collectors,
        )
    except Exception as exc:  # Final guard for future collectors and platform APIs.
        return _fallback_inventory(system_name, type(exc).__name__)


def _collect_host_inventory(
    *,
    command_runner: CommandRunner | None,
    text_reader: TextReader | None,
    system_name: str | None,
    accelerator_collectors: tuple[AcceleratorCollector, ...],
) -> dict[str, object]:
    """Collect inventory details while retaining individual section failures."""
    runner = command_runner or _default_command_runner
    reader = text_reader or _read_optional_text
    operating_system = system_name or platform.system()
    errors: list[dict[str, str]] = []

    def safely(section: str, collector: Callable[[], object], fallback: object) -> object:
        try:
            return collector()
        except Exception as exc:  # Inventory must never block a benchmark.
            errors.append({"section": section, "reason": type(exc).__name__})
            return fallback

    probes, devices = _collect_accelerators(
        accelerator_collectors,
        runner,
        errors,
    )
    inventory: dict[str, object] = {
        "schema_version": 1,
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "operating_system": {
            "system": operating_system,
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "cpu": safely("cpu", lambda: _collect_cpu(operating_system, reader, runner), {}),
        "memory": safely("memory", lambda: _collect_memory(operating_system, reader, runner), {}),
        "storage": safely(
            "storage",
            lambda: _collect_storage(operating_system, runner),
            {"filesystems": [], "block_devices": []},
        ),
        "network_interfaces": safely(
            "network_interfaces",
            lambda: _collect_network_interfaces(operating_system, reader),
            [],
        ),
        "accelerators": {
            "devices": devices,
            "probes": probes,
        },
    }
    if errors:
        inventory["collection_errors"] = errors
    return inventory


def _collect_accelerators(
    collectors: tuple[AcceleratorCollector, ...],
    command_runner: CommandRunner,
    errors: list[dict[str, str]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Collect valid vendor probes without letting plugin failures escape."""
    probes: list[dict[str, object]] = []
    devices: list[dict[str, object]] = []
    try:
        registered_collectors = tuple(collectors)
    except Exception as exc:
        errors.append({"section": "accelerator", "reason": type(exc).__name__})
        fallback = _failed_probe("unknown", "collector_registry", "collector_exception")
        return [_probe_payload(fallback)], []

    for collector in registered_collectors:
        fallback = _failed_probe("unknown", type(collector).__name__, "collector_exception")
        try:
            result = collector.collect(command_runner)
            probe, collector_devices = _validated_accelerator_result(result)
        except Exception as exc:
            errors.append({"section": "accelerator", "reason": type(exc).__name__})
            probe = _probe_payload(fallback)
            collector_devices = []
        probes.append(probe)
        devices.extend(collector_devices)
    return probes, devices


def _validated_accelerator_result(
    result: object,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Validate one plugin result before merging it into the JSON report."""
    if not isinstance(result, AcceleratorProbeResult):
        raise TypeError("accelerator collector must return AcceleratorProbeResult")
    if not isinstance(result.devices, list) or not all(
        isinstance(device, dict) for device in result.devices
    ):
        raise TypeError("accelerator collector devices must be a list of dictionaries")
    probe = _probe_payload(result)
    json.dumps({"probe": probe, "devices": result.devices})
    return probe, result.devices


def _probe_payload(result: AcceleratorProbeResult) -> dict[str, object]:
    """Create one report-safe accelerator probe record."""
    return {
        "collector": result.collector,
        "vendor": result.vendor,
        "tool": result.tool,
        "state": result.state,
        "reason": result.reason,
    }


def _fallback_inventory(system_name: str | None, reason: str) -> dict[str, object]:
    """Return the minimum JSON-safe inventory after an unexpected top-level error."""
    return {
        "schema_version": 1,
        "collected_at": None,
        "operating_system": {
            "system": system_name if isinstance(system_name, str) else "unknown",
            "release": None,
            "machine": None,
        },
        "cpu": {},
        "memory": {},
        "storage": {"filesystems": [], "block_devices": []},
        "network_interfaces": [],
        "accelerators": {"devices": [], "probes": []},
        "collection_errors": [{"section": "inventory", "reason": reason}],
    }


def _collect_cpu(
    system_name: str,
    text_reader: TextReader,
    command_runner: CommandRunner,
) -> dict[str, object]:
    """Collect CPU architecture, model, and core counts using OS-native sources."""
    cpu: dict[str, object] = {
        "architecture": platform.machine(),
        "logical_cores": os.cpu_count(),
        "physical_cores": None,
        "model": None,
        "frequency_mhz": None,
    }
    if system_name == "Linux":
        content = text_reader("/proc/cpuinfo") or ""
        values = _proc_cpuinfo_values(content)
        cpu["model"] = values.get("model name") or values.get("Hardware")
        cpu["physical_cores"] = _integer_or_none(values.get("cpu cores"))
        cpu["frequency_mhz"] = _float_or_none(values.get("cpu MHz"))
    elif system_name == "Darwin":
        cpu["model"] = _command_scalar(command_runner, ["sysctl", "-n", "machdep.cpu.brand_string"])
        cpu["physical_cores"] = _integer_or_none(
            _command_scalar(command_runner, ["sysctl", "-n", "hw.physicalcpu"])
        )
        cpu["logical_cores"] = _integer_or_none(
            _command_scalar(command_runner, ["sysctl", "-n", "hw.logicalcpu"])
        ) or cpu["logical_cores"]
    return cpu


def _collect_memory(
    system_name: str,
    text_reader: TextReader,
    command_runner: CommandRunner,
) -> dict[str, object]:
    """Collect installed and currently available system memory in bytes."""
    if system_name == "Linux":
        values = _meminfo_values(text_reader("/proc/meminfo") or "")
        return {
            "total_bytes": _kib_to_bytes(values.get("MemTotal")),
            "available_bytes": _kib_to_bytes(values.get("MemAvailable")),
            "swap_total_bytes": _kib_to_bytes(values.get("SwapTotal")),
            "swap_free_bytes": _kib_to_bytes(values.get("SwapFree")),
        }
    if system_name == "Darwin":
        return {
            "total_bytes": _integer_or_none(
                _command_scalar(command_runner, ["sysctl", "-n", "hw.memsize"])
            ),
            "available_bytes": None,
            "swap_total_bytes": None,
            "swap_free_bytes": None,
        }
    return {
        "total_bytes": None,
        "available_bytes": None,
        "swap_total_bytes": None,
        "swap_free_bytes": None,
    }


def _collect_storage(
    system_name: str,
    command_runner: CommandRunner,
) -> dict[str, object]:
    """Collect root capacity and Linux block-device static details when available."""
    usage = shutil.disk_usage("/")
    block_devices = []
    if system_name == "Linux":
        result = _run_optional_command(
            command_runner,
            ["lsblk", "--json", "--bytes", "--output", "NAME,TYPE,SIZE,MODEL,ROTA,TRAN"],
        )
        if result is not None and not result.returncode:
            try:
                payload = json.loads(result.stdout)
                block_devices = _linux_block_devices(payload)
            except (TypeError, ValueError, json.JSONDecodeError):
                block_devices = []
    return {
        "filesystems": [{
            "mount": "/",
            "total_bytes": usage.total,
            "used_bytes": usage.used,
            "free_bytes": usage.free,
        }],
        "block_devices": block_devices,
    }


def _collect_network_interfaces(
    system_name: str,
    text_reader: TextReader,
) -> list[dict[str, object]]:
    """Collect interface names and link attributes without MAC or IP addresses."""
    interfaces = []
    for _, name in socket.if_nameindex():
        entry: dict[str, object] = {
            "name": name,
            "mtu": None,
            "state": None,
            "speed_mbps": None,
            "driver": None,
        }
        if system_name == "Linux":
            base = f"/sys/class/net/{name}"
            entry["mtu"] = _integer_or_none(text_reader(f"{base}/mtu"))
            entry["state"] = _strip_or_none(text_reader(f"{base}/operstate"))
            entry["speed_mbps"] = _integer_or_none(text_reader(f"{base}/speed"))
            entry["driver"] = _linux_interface_driver(text_reader(f"{base}/device/uevent"))
        interfaces.append(entry)
    return interfaces



def _linux_block_devices(payload: object) -> list[dict[str, object]]:
    """Extract top-level physical Linux block devices from lsblk JSON output."""
    if not isinstance(payload, dict):
        return []
    entries = payload.get("blockdevices")
    if not isinstance(entries, list):
        return []
    devices = []
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("type") != "disk":
            continue
        name = _strip_or_none(entry.get("name"))
        if name is None:
            continue
        devices.append({
            "name": name,
            "size_bytes": _integer_or_none(str(entry.get("size", ""))),
            "model": _strip_or_none(entry.get("model")),
            "rotational": _integer_or_none(str(entry.get("rota", ""))),
            "transport": _strip_or_none(entry.get("tran")),
        })
    return devices


def _default_command_runner(command: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a bounded fixed command without a shell for static inventory only."""
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=5.0,
        check=False,
    )


def _run_optional_command(
    command_runner: CommandRunner,
    command: list[str],
) -> subprocess.CompletedProcess[str] | None:
    """Return None when an optional vendor tool is not installed or runnable."""
    try:
        return command_runner(command)
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None


def _read_optional_text(path: str) -> str | None:
    """Read a small procfs or sysfs file, returning None when unavailable."""
    try:
        return Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None


def _command_scalar(command_runner: CommandRunner, command: list[str]) -> str | None:
    """Return one successful command value without reporting stderr content."""
    result = _run_optional_command(command_runner, command)
    if result is None or result.returncode:
        return None
    return _strip_or_none(result.stdout)


def _unavailable_probe(vendor: str, tool: str) -> AcceleratorProbeResult:
    """Build a non-error result for an optional tool absent from the host."""
    return AcceleratorProbeResult("{}_probe".format(vendor), vendor, tool, "unavailable", [], "tool_unavailable")


def _failed_probe(vendor: str, tool: str, reason: str) -> AcceleratorProbeResult:
    """Build a safe failed collector result without recording command output."""
    return AcceleratorProbeResult("{}_probe".format(vendor), vendor, tool, "failed", [], reason)



def _linux_interface_driver(content: str | None) -> str | None:
    """Extract the kernel driver name from a Linux interface uevent file."""
    if not content:
        return None
    for line in content.splitlines():
        if line.startswith("DRIVER="):
            return _strip_or_none(line.partition("=")[2])
    return None


def _proc_cpuinfo_values(content: str) -> dict[str, str]:
    """Extract the first Linux CPU record's key/value fields."""
    values = {}
    for line in content.splitlines():
        if not line.strip():
            break
        if ":" in line:
            key, value = line.split(":", 1)
            values[key.strip()] = value.strip()
    return values


def _meminfo_values(content: str) -> dict[str, str]:
    """Extract Linux /proc/meminfo key/value fields."""
    values = {}
    for line in content.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip().split()[0]
    return values


def _kib_to_bytes(value: str | None) -> int | None:
    """Convert a decimal KiB string to bytes when valid."""
    parsed = _integer_or_none(value)
    return parsed * 1024 if parsed is not None else None


def _integer_or_none(value: str | None) -> int | None:
    """Convert one text value to an integer when it contains only digits."""
    if value is None:
        return None
    try:
        return int(value.strip())
    except (AttributeError, ValueError):
        return None



def _float_or_none(value: str | None) -> float | None:
    """Convert one text value to a floating-point number when valid."""
    if value is None:
        return None
    try:
        return float(value.strip())
    except (AttributeError, ValueError):
        return None


def _integer_or_text(value: str) -> int | str:
    """Preserve an unrecognized tool value while normalizing integer fields."""
    parsed = _integer_or_none(value)
    return parsed if parsed is not None else value


def _strip_or_none(value: str | None) -> str | None:
    """Trim one optional string, returning None for empty values."""
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _strip_device_prefix(value: str) -> str:
    """Remove a generic device label while retaining the vendor model text."""
    return re.sub(r"^(?:GPU|Device)\s*\d+\s*[:=-]?\s*", "", value, flags=re.IGNORECASE)
