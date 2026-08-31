"""Unit tests for sanitized, extensible host hardware inventory collection."""

from __future__ import annotations

import subprocess

from benchmark import host_inventory


def _runner(command: list[str]) -> subprocess.CompletedProcess[str]:
    """Return deterministic vendor CLI output without probing local hardware."""
    outputs = {
        (
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ): "0, NVIDIA H100, 550.54.15, 81559\n",
        ("npu-smi", "info"): (
            "Driver Version : 24.1\n"
            "| 0 | Ascend 910B | Healthy |\n"
            "| 1 | Ascend 910B | Healthy |\n"
        ),
        ("mx-smi", "-L"): "GPU 0: MetaX C550\n",
    }
    output = outputs.get(tuple(command))
    if output is None:
        raise FileNotFoundError(command[0])
    return subprocess.CompletedProcess(command, 0, stdout=output, stderr="ignored")


def _text_reader(path: str) -> str | None:
    """Provide deterministic Linux procfs and sysfs content to the collector."""
    values = {
        "/proc/cpuinfo": "model name\t: Test CPU\ncpu cores\t: 32\n\n",
        "/proc/meminfo": "MemTotal:       1048576 kB\nMemAvailable:    524288 kB\n",
        "/sys/class/net/eth0/mtu": "9000\n",
        "/sys/class/net/eth0/operstate": "up\n",
        "/sys/class/net/eth0/speed": "100000\n",
    }
    return values.get(path)


def test_collect_host_inventory_merges_vendor_collectors_without_identifiers(monkeypatch) -> None:
    """Report static multi-vendor hardware details without MACs, UUIDs, or stderr."""
    monkeypatch.setattr(host_inventory.socket, "if_nameindex", lambda: [(1, "eth0")])

    inventory = host_inventory.collect_host_inventory(
        command_runner=_runner,
        text_reader=_text_reader,
        system_name="Linux",
    )

    assert inventory["schema_version"] == 1
    assert inventory["cpu"] == {
        "architecture": host_inventory.platform.machine(),
        "logical_cores": host_inventory.os.cpu_count(),
        "physical_cores": 32,
        "model": "Test CPU",
        "frequency_mhz": None,
    }
    assert inventory["memory"] == {
        "total_bytes": 1048576 * 1024,
        "available_bytes": 524288 * 1024,
        "swap_total_bytes": None,
        "swap_free_bytes": None,
    }
    assert inventory["network_interfaces"] == [{
        "name": "eth0",
        "mtu": 9000,
        "state": "up",
        "speed_mbps": 100000,
        "driver": None,
    }]
    devices = inventory["accelerators"]["devices"]
    assert devices == [
        {
            "vendor": "nvidia",
            "index": 0,
            "name": "NVIDIA H100",
            "driver_version": "550.54.15",
            "memory_total_mib": 81559,
        },
        {
            "vendor": "huawei",
            "index": 0,
            "name": "Ascend 910B",
            "driver_version": "24.1",
        },
        {
            "vendor": "huawei",
            "index": 1,
            "name": "Ascend 910B",
            "driver_version": "24.1",
        },
        {"vendor": "muxi", "index": 0, "name": "MetaX C550"},
    ]
    assert all("mac_address" not in interface for interface in inventory["network_interfaces"])
    assert all("uuid" not in device for device in devices)


def test_missing_or_failing_vendor_tool_is_reported_without_failing_inventory(monkeypatch) -> None:
    """Treat optional accelerator tooling as unavailable rather than a benchmark error."""
    monkeypatch.setattr(host_inventory.socket, "if_nameindex", lambda: [])

    inventory = host_inventory.collect_host_inventory(
        command_runner=lambda _command: (_ for _ in ()).throw(FileNotFoundError()),
        text_reader=lambda _path: None,
        system_name="Linux",
    )

    probes = inventory["accelerators"]["probes"]
    assert all(probe["state"] == "unavailable" for probe in probes)
    assert inventory["accelerators"]["devices"] == []
    assert "collection_errors" not in inventory


def test_custom_accelerator_collector_can_extend_inventory(monkeypatch) -> None:
    """Allow a future vendor plugin without changing the host collector core."""
    monkeypatch.setattr(host_inventory.socket, "if_nameindex", lambda: [])

    class _CustomCollector:
        def collect(self, _command_runner):
            return host_inventory.AcceleratorProbeResult(
                collector="custom_probe",
                vendor="custom",
                tool="custom-smi",
                state="available",
                devices=[{"vendor": "custom", "name": "Custom X1"}],
            )

    inventory = host_inventory.collect_host_inventory(
        command_runner=lambda _command: (_ for _ in ()).throw(AssertionError()),
        text_reader=lambda _path: None,
        system_name="Linux",
        accelerator_collectors=(_CustomCollector(),),
    )

    assert inventory["accelerators"] == {
        "devices": [{"vendor": "custom", "name": "Custom X1"}],
        "probes": [{
            "collector": "custom_probe",
            "vendor": "custom",
            "tool": "custom-smi",
            "state": "available",
            "reason": None,
        }],
    }


def test_malformed_custom_collector_is_isolated_without_stopping_inventory(monkeypatch) -> None:
    """Treat an invalid extension result as a non-fatal accelerator probe failure."""
    monkeypatch.setattr(host_inventory.socket, "if_nameindex", lambda: [])

    class _MalformedCollector:
        def collect(self, _command_runner):
            return host_inventory.AcceleratorProbeResult(
                collector="invalid_probe",
                vendor="custom",
                tool="custom-smi",
                state="available",
                devices=None,
            )

    inventory = host_inventory.collect_host_inventory(
        command_runner=lambda _command: (_ for _ in ()).throw(AssertionError()),
        text_reader=lambda _path: None,
        system_name="Linux",
        accelerator_collectors=(_MalformedCollector(),),
    )

    assert inventory["accelerators"] == {
        "devices": [],
        "probes": [{
            "collector": "unknown_probe",
            "vendor": "unknown",
            "tool": "_MalformedCollector",
            "state": "failed",
            "reason": "collector_exception",
        }],
    }
    assert {"section": "accelerator", "reason": "TypeError"} in inventory["collection_errors"]
    assert inventory["cpu"]["model"] is None


def test_unexpected_inventory_error_returns_minimum_safe_payload(monkeypatch) -> None:
    """Keep benchmark reporting available when an outer platform operation fails."""
    monkeypatch.setattr(
        host_inventory.platform,
        "release",
        lambda: (_ for _ in ()).throw(OSError("platform unavailable")),
    )

    inventory = host_inventory.collect_host_inventory(
        command_runner=lambda _command: (_ for _ in ()).throw(FileNotFoundError()),
        text_reader=lambda _path: None,
        system_name="Linux",
        accelerator_collectors=(),
    )

    assert inventory == {
        "schema_version": 1,
        "collected_at": None,
        "operating_system": {"system": "Linux", "release": None, "machine": None},
        "cpu": {},
        "memory": {},
        "storage": {"filesystems": [], "block_devices": []},
        "network_interfaces": [],
        "accelerators": {"devices": [], "probes": []},
        "collection_errors": [{"section": "inventory", "reason": "OSError"}],
    }
