#!/usr/bin/env python3
"""
LLM Prefill / Decode 性能基准测试工具 (benchmark.py)
=====================================================

通用 LLM 推理性能基准测试工具，适用于任意模型和两种请求后端。
可测量 Prefill 吞吐、Decode 生成速度、TTFT、TPOT、预估 ITL、QPS、Goodput，
并支持混合负载、并发扫描、PD 分离比例计算、测试套件和 JSON 报告。

重要的长度语义：
  每次测试实际发送的 Prompt 长度和输出上限都以生成后的 DatasetBatch 为准，
  不要把 CLI 默认的 context_len/max_tokens 当作随机数据集的实际长度。
  控制台和报告会分别记录“请求输出上限”（发送给后端的 max_tokens）以及
  “实际平均生成输出”（服务端真实返回的 completion token 数）。

  1. 数据集
     - text（默认）：使用中文填充文本；输入长度由 --context-len 控制，输出上限
       由 --max-tokens/--output-len 控制。
     - random：使用可复现的合成 token 序列，需要 NumPy；通过 --random-input-len、
       --random-output-len、--random-prefix-len、--random-range-ratio 和
       --random-seed 独立控制随机负载。长度统计以实际采样和 tokenizer 重编码结果为准。

  random 数据集的共享前缀由 --random-prefix-len 定义，所有请求可共享该合成前缀；
  --share-prefix、--prefix-ratio 和 --random-shared-prefix 仅适用于 text 数据集。
  对 text 数据集，--share-prefix 开启后按 --prefix-ratio 共享前缀；关闭时每个请求
  使用独立前缀以尽量避免 Cache 命中。

  2. 测量模式与请求控制
     - offline : 直接用 vLLM LLM 类测量，需本地 GPU；--tp-size 和 --gpu-mem-util
       控制离线引擎配置。
     - api     : 通过已启动的 OpenAI 兼容服务测量 TTFT/TPOT；--api-base、--model、
       --tokenizer 和 --api-key 控制服务连接，向非 HTTPS 远端发送 Key 需显式使用
       --allow-insecure-api-key。
     --concurrency/-c       : 并发请求数
     --num-prompts/-n       : 总请求数，未指定时默认等于并发数
     --no-warmup            : 跳过 sweep/SLO 的正式 workload 预热，以及其他场景的预热
     --ignore-eos/--no-ignore-eos : 控制 EOS 是否允许提前结束

  3. 预设测试用例 (--preset)
     prefill-max     : 测试系统最大 Prefill 吞吐能力
     decode-max      : 测试系统最大 Decode 生成能力
     mixed           : 模拟真实混合负载 (Prefill + Decode)
     prefill-sweep   : 自动搜索最大 Prefill 吞吐量 (逐步增加并发)
     decode-sweep    : 自动搜索最大 Decode 吞吐量 (逐步增加并发)
     pd-ratio        : 自动测量 P/D 吞吐量并计算最佳 PD 分离比例
     mixed-workload  : 混合负载测试 (不同长短上下文组合)

  sweep 由实际 DatasetBatch 的输出上限决定：所有请求上限为 1 才是 Prefill，
  所有请求上限大于 1 才是 Decode；同一轮混合两种阶段会被拒绝，避免吞吐统计失真。

  统计口径：收到首个内容 token 的请求才视为成功；TTFT 从请求发出到首个内容
  token 到达，TPOT 仅由两个不同内容 token 的到达时刻计算。Prompt 全流程工作负载率
  和整体交付吞吐量使用完整测试窗口；Prefill 阶段吞吐量使用所有成功请求
  [发送, 首 token] 区间的并集，Decode 阶段吞吐量使用 [首 token, 最后内容 token]
  区间的并集，且 Decode 分子为输出 token 数减去首 token。阶段活动窗口会排除
  所有阶段均不活跃的测试间隙，但仍包含客户端可观测的排队与网络延迟；它不是服务端
  scheduler 内部纯计算时间。若任一成功请求的输出 token 数无法由服务端 usage 或
  本地 tokenizer 验证，则涉及输出 token 的吞吐量不报告。Goodput 的分母始终是全部请求，且仅当 TTFT ≤ --slo-ttft
  和（输出超过 1 token 时）TPOT ≤ --slo-tpot 才计入达标。

  容量与吞吐扫描：
     --preset prefill-sweep / decode-sweep：以 1、2、4、… 的并发进行吞吐扫描；
       对合格且指标可验证的档位，吞吐相对当前最佳值提升不足 5% 时提前停止。
       Prefill 扫描要求输出上限为 1；Decode 扫描要求输出上限大于 1。
     --preset slo-capacity-sweep：预热、粗扫、binary/linear 精扫和候选确认均从同一
       数据集实例按轮生成新的请求 batch。random 数据集会缓存首次生成的共享前缀，
       因此设置相同随机种子时前缀稳定，而每轮会推进随机状态以生成不同后缀。
       首次出现 SLO 不通过的并发后，可选 linear 或 binary-confirm 精扫。linear 默认逐并发
       测试，也可通过 --slo-capacity-linear-step 设置采样步长；步长大于 1 时返回最高已采样
       且通过的并发。在区间内首次发现新的失败点后，仍逐并发额外测试
       --slo-capacity-confirm-window 个后续并发来确认边界；linear 和 binary-confirm 都会将最终候选
       重复测量至 --slo-capacity-confirm-rounds 指定的总轮数，任一确认轮失败则回退到下一个
       已通过候选。默认 binary-confirm 先二分定位候选，再检查候选两侧的确认窗口并重复测量候选。通过
       --slo-capacity-search-strategy、--slo-capacity-linear-step、
       --slo-capacity-confirm-window 和 --slo-capacity-confirm-rounds 配置。JSON 的 warmup.rounds 与
       warmup.requests_per_round 可分别控制预热轮数和每轮请求数；后者默认 1，避免长上下文
       预热被正式扫描的最小批量规则放大。每轮以 TTFT/TPOT 判定请求是否达标；SLO
       通过率须达到 --min-goodput-pct（未设置或为 0 时保持 100% 严格行为），且失败率
       不超过 --max-failure-rate，才视为该并发通过。容量探针失败用于定位边界，最终质量
       门禁仅校验选出的最大通过并发。

     提供两份 128K 输入 / 2K 输出默认配置，扫描数据集均为 RandomDataset：
       benchmark-config-throughput-sweep-128k-2k.json：包含最大 Prefill 与
       最大 Decode 吞吐扫描；固定使用 131072 输入、2048 输出的随机 token 序列，
       其中 Prefill case 显式将输出（含 random_output_len）覆盖为 1 token。
       benchmark-config-slo-capacity-128k-2k-cache-hit-0.7.json：使用 91750 tokens
       共享前缀 + 39322 tokens 独有输入的 RandomDataset，模拟约 70% 前缀缓存命中
       场景，搜索严格 SLO 下的最大并发，默认 TTFT ≤ 60 秒、TPOT ≤ 50ms（0.05 秒）。
       RandomDataset 依赖 NumPy；两份配置中的服务地址、模型和 tokenizer 均为本地
       模板值，应按实际部署修改。

  4. 配置化测试套件与 JSON 报告
     --config FILE          : 从 JSON 文件读取 defaults/cases/matrix/repeat
     --report FILE          : 指定 JSON 报告路径
     --case / --tag         : 筛选测试用例
     --validate-config      : 只校验配置，不发送请求
     --list-cases           : 展开并列出 matrix/repeat 后的全部用例
     报告中的 workload 会保存实际 DatasetBatch 的输入/输出长度统计、共享前缀、
     独有 Prompt 长度和 has_decode；API 服务器资源只统计测试窗口内、且可观测到运行/排队
     请求的活跃样本（后端未暴露该指标时回退到所有窗口样本），包含 min/avg/max 与
     P50/P90/P95/P99，不在测试结束后额外抓取窗口外快照。

  5. 依赖
     pip install vllm transformers requests torch
     random 数据集额外需要 NumPy（建议在 benchmark 环境中固定版本）：
     pip install numpy

  6. 其他
     --slo-ttft / --slo-tpot : Goodput 使用的 TTFT/TPOT 阈值
     --max-failure-rate     : 配置套件允许的最大失败率
     --min-goodput-pct      : 配置套件要求的最低 Goodput 百分比
     --seed                 : 混合负载分配的随机种子

用法示例：
  # ── 基础用法 ──────────────────────────────────────────────────────

  # 使用预设快速测试 Prefill 吞吐
  python benchmark.py --mode api --preset prefill-max --api-base http://localhost:8001/v1

  # 使用预设快速测试 Decode 吞吐
  python benchmark.py --mode api --preset decode-max --api-base http://localhost:8001/v1

  # ── 自定义参数 ────────────────────────────────────────────────────

  # 4 并发发送 20 个请求，128K 上下文，独立前缀 (测纯算力 Prefill)
  python benchmark.py --mode api --context-len 131072 -c 4 -n 20 \\
      --api-base http://localhost:8001/v1

  # 8 并发，短输入 + 长输出 (测 Decode 能力)
  python benchmark.py --mode api --context-len 128 --max-tokens 4096 -c 8 \\
      --api-base http://localhost:8001/v1

  # 指定模型名 (当服务端注册名与默认不同时)
  python benchmark.py --mode api --model my-model-name \\
      --tokenizer /data/models/my-model --api-base http://localhost:8001/v1

  # ── 前缀共享 ──────────────────────────────────────────────────────

  # 80% 共享前缀，测前缀缓存命中下的 TTFT
  python benchmark.py --mode api --context-len 131072 -c 4 -n 20 \\
      --share-prefix --prefix-ratio 0.8 --api-base http://localhost:8001/v1

  # RandomDataset：固定共享前缀和输出上限（random 不读取 --share-prefix/--prefix-ratio）
  python benchmark.py --mode api --dataset random --random-input-len 39322 --random-prefix-len 91750 --random-output-len 8192 --random-range-ratio 0 --random-seed 42 --api-base http://localhost:8001/v1

  # RandomDataset：输入/输出长度浮动，报告显示实际 min/max/avg
  python benchmark.py --mode api --dataset random --random-input-len 8192 --random-output-len 1024 --random-range-ratio 0.1,0.2 --random-seed 42 --api-base http://localhost:8001/v1 -c 8 -n 32

  # ── 高级场景 ──────────────────────────────────────────────────────

  # 自动搜索最大 Prefill 吞吐 (逐步增加并发直到性能饱和)
  python benchmark.py --mode api --preset prefill-sweep --api-base http://localhost:8001/v1

  # 计算最佳 PD 分离比例
  python benchmark.py --mode api --preset pd-ratio --api-base http://localhost:8001/v1

  # 混合负载测试
  python benchmark.py --mode api --preset mixed-workload --api-base http://localhost:8001/v1

  # 自定义混合负载 (30% 短入长出 + 50% 中等 + 20% 长入短出)
  python benchmark.py --mode api -c 8 -n 32 \\
      --workload-mix '128:4096:0.3,4096:1024:0.5,131072:1:0.2' \\
      --api-base http://localhost:8001/v1

  # ── 配置化测试套件 ────────────────────────────────────────────────

  # 校验并查看示例配置展开后的用例
  python benchmark.py --config benchmark-config.example.json --validate-config
  python benchmark.py --config benchmark-config.example.json --list-cases

  # 运行完整套件并输出 JSON 报告；也可按 tag/case 筛选
  python benchmark.py --config benchmark-config.example.json --report results.json
  python benchmark.py --config benchmark-config.example.json --tag smoke
  python benchmark.py --config benchmark-config.example.json --case 'decode-*'

  # 配置文件中的 SLO 示例（TTFT 单位为秒，TPOT 单位为秒；50ms = 0.05）
  {
    "defaults": {
      "slo": {
        "ttft_seconds": 60.0,
        "tpot_seconds": 0.05
      },
      "min_goodput_pct": 90.0,
      "max_failure_rate": 0.0
    }
  }
  # 也可以在命令行模式使用：--slo-ttft 60 --slo-tpot 0.05

  # ── 云端 API ──────────────────────────────────────────────────────

  # 测试云端 API 服务 (需要 API Key)
  python benchmark.py --mode api --model qwen-plus \\
      --api-base https://dashscope.aliyuncs.com/compatible-mode/v1 \\
      --api-key $API_KEY --context-len 8192 -c 4

  # ── Nsys Profiling ────────────────────────────────────────────────

  # 配合 docker-start-deepseek.sh 的 ENABLE_NSYS=true 使用。
  # 设置环境变量后，benchmark 会在测试开始前自动触发 nsys start，
  # 测试结束后自动触发 nsys stop，精确采集 benchmark 流量的 profile。
  #
  # 环境变量：
  #   NSYS_PROFILE          - 启用 nsys 控制 (true/false, 默认 false)
  #   NSYS_CONTAINER        - 推理容器名 (默认: deepseek-v4-flash)
  #   NSYS_SESSION          - nsys session 名 (默认: vllm)
  #
  # 示例：
  NSYS_PROFILE=true NSYS_CONTAINER=deepseek-v4-flash \\
      python benchmark.py --mode api --preset prefill-max
"""

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import math
import os
import platform
import random
try:
    import requests
except ImportError:
    requests = None
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime
from typing import Any
from urllib.parse import urlparse

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
try:
    # Installed package / console-script path.
    from .benchmark_datasets import (
        BenchmarkDataset,
        DatasetBatch,
        TextDataset,
        create_dataset,
        parse_range_ratio,
    )
except ImportError:
    # Direct execution: python benchmark/benchmark.py ...
    from benchmark_datasets import (
        BenchmarkDataset,
        DatasetBatch,
        TextDataset,
        create_dataset,
        parse_range_ratio,
    )


# ── Nsys Profiling Control ─────────────────────────────────────────────

# Controlled via environment variables:
#   NSYS_PROFILE    = true/false (default: false)
#   NSYS_CONTAINER  = container name (default: deepseek-v4-flash)
#   NSYS_SESSION    = nsys session name (default: vllm)

NSYS_PROFILE = os.environ.get("NSYS_PROFILE", "false").lower() == "true"
NSYS_CONTAINER = os.environ.get("NSYS_CONTAINER", "deepseek-v4-flash")
NSYS_SESSION = os.environ.get("NSYS_SESSION", "vllm")
NSYS_OUTPUT_NAME = os.environ.get("NSYS_OUTPUT_NAME", "profile")


def require_requests():
    """Fail with an actionable message only when an HTTP benchmark is executed."""
    if requests is None:
        raise RuntimeError("API benchmark mode requires the 'requests' package: pip install requests")


def nsys_start():
    """Trigger nsys to start collecting profile data in the inference container."""
    if not NSYS_PROFILE:
        return
    cmd = ["docker", "exec", NSYS_CONTAINER, "nsys", "start", f"--session={NSYS_SESSION}",
           "-o", f"/nsys-output/{NSYS_OUTPUT_NAME}"]
    print(f"  [nsys] Starting profiling: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"  [nsys] Profiling started successfully")
        else:
            print(f"  [nsys] WARNING: nsys start returned code {result.returncode}")
            if result.stderr.strip():
                print(f"         {result.stderr.strip()}")
    except subprocess.TimeoutExpired:
        print("  [nsys] WARNING: nsys start timed out")
    except Exception as e:
        print(f"  [nsys] WARNING: failed to start profiling: {e}")


def nsys_stop():
    """Trigger nsys to stop collecting and save the profile data."""
    if not NSYS_PROFILE:
        return
    cmd = ["docker", "exec", NSYS_CONTAINER, "nsys", "stop", f"--session={NSYS_SESSION}"]
    print(f"  [nsys] Stopping profiling: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print(f"  [nsys] Profiling stopped, .nsys-rep saved")
        else:
            print(f"  [nsys] WARNING: nsys stop returned code {result.returncode}")
            if result.stderr.strip():
                print(f"         {result.stderr.strip()}")
    except subprocess.TimeoutExpired:
        print("  [nsys] WARNING: nsys stop timed out (profile may still be saving)")
    except Exception as e:
        print(f"  [nsys] WARNING: failed to stop profiling: {e}")


# ── System Environment Info ────────────────────────────────────────────

def collect_and_print_system_info(args):
    """
    Collect and print system environment info for benchmark analysis.
    Includes client-side info and server-side info (queried via API).
    """
    print("=" * 70)
    print("  系统环境信息")
    print("=" * 70)

    # ── Client (benchmark runner) info ──
    print("  ── 客户端 (Benchmark Runner) ──")
    print(f"    时间戳          : {datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"    Python          : {platform.python_version()}")
    print(f"    OS              : {platform.system()} {platform.release()} ({platform.machine()})")

    # CPU info
    try:
        if platform.system() == "Linux":
            cpu_info = subprocess.check_output("lscpu | grep 'Model name'", shell=True, text=True).strip()
            cpu_name = cpu_info.split(":")[-1].strip()
            cpu_count = os.cpu_count()
            print(f"    CPU             : {cpu_name} ({cpu_count} cores)")
        elif platform.system() == "Darwin":
            cpu_name = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
            cpu_count = os.cpu_count()
            print(f"    CPU             : {cpu_name} ({cpu_count} cores)")
        else:
            print(f"    CPU cores       : {os.cpu_count()}")
    except Exception:
        print(f"    CPU cores       : {os.cpu_count()}")

    # Memory info
    try:
        if platform.system() == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        mem_kb = int(line.split()[1])
                        print(f"    RAM             : {mem_kb / 1024 / 1024:.1f} GB")
                        break
        elif platform.system() == "Darwin":
            mem_bytes = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip())
            print(f"    RAM             : {mem_bytes / 1024**3:.1f} GB")
    except Exception:
        pass

    # ── Server info (via API) ──
    api_base = getattr(args, "api_base", "http://localhost:8000/v1")
    # Properly strip /v1 suffix (rstrip would strip individual chars, breaking port numbers)
    if api_base.endswith("/v1/"):
        base_url = api_base[:-4]
    elif api_base.endswith("/v1"):
        base_url = api_base[:-3]
    else:
        base_url = api_base
    base_url = base_url.rstrip("/")
    print()
    print(f"  ── 服务端 (API: {api_base}) ──")

    # Auto-detect server type: vLLM vs SGLang
    server_type = "unknown"
    server_version = None

    # Method 1: Try vLLM /version
    try:
        resp = requests.get(f"{base_url}/version", timeout=5)
        if resp.status_code == 200:
            version_data = resp.json()
            server_version = version_data.get("version", "unknown")
            server_type = "vLLM"
    except Exception:
        pass

    # Method 2: Try SGLang /get_server_info
    if server_type == "unknown":
        try:
            resp = requests.get(f"{base_url}/get_server_info", timeout=5)
            if resp.status_code == 200:
                server_type = "SGLang"
                sglang_info = resp.json()
                server_version = sglang_info.get("version", None)
        except Exception:
            pass

    # Method 3: Check /metrics for vllm: or sglang: metric prefixes (very reliable)
    if server_type == "unknown":
        try:
            resp = requests.get(f"{base_url}/metrics", timeout=5)
            if resp.status_code == 200:
                # Just check the first ~2000 chars for speed
                sample = resp.text[:2000]
                if "vllm:" in sample or "vllm_" in sample:
                    server_type = "vLLM"
                elif "sglang:" in sample or "sglang_" in sample:
                    server_type = "SGLang"
        except Exception:
            pass

    # Method 4: Try /v1/config (vLLM-specific endpoint)
    if server_type == "unknown":
        try:
            resp = requests.get(f"{base_url}/v1/config", timeout=5)
            if resp.status_code == 200:
                server_type = "vLLM"
        except Exception:
            pass

    # Method 5: Check /health response headers
    if server_type == "unknown":
        try:
            resp = requests.get(f"{base_url}/health", timeout=5)
            server_header = resp.headers.get("server", "").lower()
            if "sglang" in server_header:
                server_type = "SGLang"
            elif "vllm" in server_header or "uvicorn" in server_header:
                server_type = "vLLM"
        except Exception:
            pass

    if server_version:
        print(f"    版本            : {server_version}")

    # ── Model info (shared endpoint: /v1/models) ──
    try:
        resp = requests.get(f"{api_base}/models", timeout=5)
        if resp.status_code == 200:
            models_data = resp.json()
            models = models_data.get("data", [])
            for m in models:
                model_id = m.get("id", "unknown")
                print(f"    模型            : {model_id}")
                if m.get("max_model_len"):
                    print(f"    最大上下文长度  : {m['max_model_len']}")
    except Exception:
        pass

    # ── Server-specific config ──
    if server_type == "SGLang":
        _print_sglang_server_info(base_url)
    else:
        _print_vllm_server_info(base_url, api_base)

    # ── Metrics (shared Prometheus endpoint) ──
    _print_metrics_info(base_url, server_type)

    # ── Health check ──
    try:
        resp = requests.get(f"{base_url}/health", timeout=5)
        status = "✓ 正常" if resp.status_code == 200 else f"⚠ HTTP {resp.status_code}"
        print(f"    服务状态        : {status}")
    except Exception as e:
        print(f"    服务状态        : ⚠ 不可达 ({e})")

    print("=" * 70)
    print()


def _print_vllm_server_info(base_url, api_base):
    """Print vLLM-specific server configuration."""
    try:
        resp = requests.get(f"{base_url}/v1/config", timeout=5)
        if resp.status_code == 200:
            config = resp.json()
            useful_keys = [
                ("tensor_parallel_size", "TP Size"),
                ("pipeline_parallel_size", "PP Size"),
                ("data_parallel_size", "DP Size"),
                ("max_num_seqs", "Max Num Seqs"),
                ("max_num_batched_tokens", "Max Batched Tokens"),
                ("gpu_memory_utilization", "GPU 显存利用率"),
                ("kv_cache_dtype", "KV Cache 类型"),
                ("enable_chunked_prefill", "Chunked Prefill"),
                ("enable_prefix_caching", "Prefix Caching"),
                ("attention_backend", "Attention Backend"),
            ]
            for key, label in useful_keys:
                val = config.get(key)
                if val is not None:
                    print(f"    {label:<18s}: {val}")
    except Exception:
        pass


def _print_sglang_server_info(base_url):
    """Print SGLang-specific server configuration."""
    try:
        resp = requests.get(f"{base_url}/get_server_info", timeout=5)
        if resp.status_code == 200:
            info = resp.json()
            useful_keys = [
                ("tp_size", "TP Size"),
                ("dp_size", "DP Size"),
                ("max_total_num_tokens", "Max Total Tokens"),
                ("max_prefill_tokens", "Max Prefill Tokens"),
                ("max_running_requests", "Max Running Reqs"),
                ("mem_fraction_static", "GPU 显存利用率"),
                ("chunked_prefill_size", "Chunked Prefill Size"),
                ("enable_mixed_chunk", "Mixed Chunk"),
                ("attention_backend", "Attention Backend"),
                ("sampling_backend", "Sampling Backend"),
                ("schedule_policy", "调度策略"),
            ]
            for key, label in useful_keys:
                val = info.get(key)
                if val is not None:
                    print(f"    {label:<18s}: {val}")

            # Print CLI args if available
            server_args = info.get("server_args", {})
            if server_args:
                extra_keys = [
                    ("kv_cache_dtype", "KV Cache 类型"),
                    ("context_length", "Context Length"),
                    ("disable_radix_cache", "Radix Cache 禁用"),
                ]
                for key, label in extra_keys:
                    val = server_args.get(key)
                    if val is not None:
                        print(f"    {label:<18s}: {val}")
    except Exception:
        pass

    # SGLang also exposes /get_model_info
    try:
        resp = requests.get(f"{base_url}/get_model_info", timeout=5)
        if resp.status_code == 200:
            model_info = resp.json()
            for key, label in [
                ("is_generation", "生成模型"),
                ("num_gpu_blocks", "GPU KV Blocks"),
            ]:
                val = model_info.get(key)
                if val is not None:
                    print(f"    {label:<18s}: {val}")
    except Exception:
        pass


def _print_metrics_info(base_url, server_type):
    """Extract GPU info from Prometheus /metrics endpoint (shared by both vLLM and SGLang)."""
    import re
    try:
        resp = requests.get(f"{base_url}/metrics", timeout=5)
        if resp.status_code != 200:
            return
        metrics_text = resp.text

        gpu_type = None
        gpu_blocks = None
        sglang_token_usage_pct = None

        for line in metrics_text.split("\n"):
            if line.startswith("#"):
                continue

            # GPU type (vLLM uses gpu_type label, SGLang may use model_name or similar)
            if not gpu_type:
                match = re.search(r'gpu_type="([^"]+)"', line)
                if match:
                    gpu_type = match.group(1)

            # vLLM reports a block count. SGLang's token_usage is instead a
            # 0..1 utilization gauge, so it must not be truncated to an int.
            if gpu_blocks is None and "num_gpu_blocks" in line:
                try:
                    val = float(line.split()[-1])
                    if val > 0:
                        gpu_blocks = int(val)
                except (ValueError, IndexError):
                    pass
            if "sglang:token_usage" in line or "sglang_token_usage" in line:
                try:
                    sglang_token_usage_pct = _metric_as_percent(float(line.split()[-1]))
                except (ValueError, IndexError):
                    pass

        if gpu_type:
            print(f"    GPU 型号        : {gpu_type}")
        if gpu_blocks is not None:
            print(f"    {'GPU KV Blocks':<18s}: {gpu_blocks}")
        if sglang_token_usage_pct is not None:
            print(f"    {'GPU KV Cache':<18s}: {sglang_token_usage_pct:.1f}%")
    except Exception:
        pass


def build_request_batch(
    args,
    tokenizer,
    num_requests: int,
    *,
    input_len: int | None = None,
    output_len: int | None = None,
    share_prefix: bool | None = None,
    random_range_ratio: str | float | tuple[float, float] | None = None,
    run_id: str = "",
    dataset: BenchmarkDataset | None = None,
    random_seed: int | None = None,
    random_generation: int | None = None,
) -> DatasetBatch:
    """Generate one workload batch through the selected dataset implementation."""
    dataset_name = args.dataset
    resolved_input_len = input_len if input_len is not None else args.context_len
    resolved_output_len = output_len if output_len is not None else args.max_tokens
    if dataset_name == "text":
        text_dataset = dataset if dataset is not None else TextDataset(random_seed=args.seed)
        if not isinstance(text_dataset, TextDataset):
            raise ValueError("text dataset selection requires a TextDataset instance")
        return text_dataset.sample(
            tokenizer,
            num_requests,
            input_len=resolved_input_len,
            output_len=resolved_output_len,
            prefix_ratio=args.prefix_ratio,
            share_prefix=args.share_prefix if share_prefix is None else share_prefix,
            random_shared_prefix=args.random_shared_prefix,
            run_id=run_id,
        )

    random_dataset = dataset if dataset is not None else create_dataset(
        dataset_name,
        random_seed=args.random_seed if args.random_seed is not None else args.seed,
    )
    random_input_len = input_len if input_len is not None else (
        args.random_input_len if args.random_input_len is not None else resolved_input_len
    )
    random_output_len = output_len if output_len is not None else (
        args.random_output_len if args.random_output_len is not None else resolved_output_len
    )
    return random_dataset.sample(
        tokenizer,
        num_requests,
        input_len=random_input_len,
        output_len=random_output_len,
        prefix_len=args.random_prefix_len,
        range_ratio=(
            args.random_range_ratio
            if random_range_ratio is None
            else random_range_ratio
        ),
    )


def _token_length_stats(lengths: list[int]) -> dict[str, int | float]:
    """Return JSON-safe shape statistics for a non-empty token-length list."""
    if not lengths:
        return {"min": 0, "max": 0, "avg": 0.0, "total": 0}
    return {
        "min": min(lengths),
        "max": max(lengths),
        "avg": sum(lengths) / len(lengths),
        "total": sum(lengths),
    }


def summarize_workload_lengths(
    dataset_name: str,
    prompt_lens: list[int],
    output_lens: list[int],
    *,
    shared_prefix_tokens: int = 0,
) -> dict[str, Any]:
    """Describe actual request shapes when batches must be combined."""
    unique_prompt_lens = [max(0, length - shared_prefix_tokens) for length in prompt_lens]
    prompt_stats = _token_length_stats(prompt_lens)
    return {
        "dataset": dataset_name,
        "request_count": len(prompt_lens),
        "prompt_tokens": prompt_stats,
        "requested_output_tokens": _token_length_stats(output_lens),
        "shared_prefix_tokens": shared_prefix_tokens,
        "shared_prefix_ratio": (
            shared_prefix_tokens / prompt_stats["avg"] if prompt_stats["avg"] else 0.0
        ),
        "unique_prompt_tokens": _token_length_stats(unique_prompt_lens),
        "has_decode": any(length > 1 for length in output_lens),
    }


def summarize_dataset_batch(batch: DatasetBatch, dataset_name: str) -> dict[str, Any]:
    """Describe the generated request batch that is actually sent to a backend.

    Requested output tokens are limits passed to the backend. They are distinct
    from generated output tokens reported by a completed benchmark round.
    """
    return summarize_workload_lengths(
        dataset_name,
        batch.prompt_lens,
        batch.output_lens,
        shared_prefix_tokens=batch.shared_prefix_len,
    )


def _format_token_stats(stats: dict[str, int | float]) -> str:
    """Format fixed or variable batch token lengths without hiding variation."""
    if stats["min"] == stats["max"]:
        return f"{stats['min']}"
    return f"平均 {stats['avg']:.1f} [{stats['min']}, {stats['max']}]"


def print_workload_summary(workload: dict[str, Any], *, indent: str = "") -> None:
    """Print runtime workload shape shared by text and random datasets."""
    print(f"{indent}数据集                      : {workload['dataset']}")
    print(f"{indent}Prompt 长度（实际请求）       : {_format_token_stats(workload['prompt_tokens'])} tokens")
    print(f"{indent}请求输出上限（实际请求）       : {_format_token_stats(workload['requested_output_tokens'])} tokens")
    print(
        f"{indent}共享前缀（实际请求）           : {workload['shared_prefix_tokens']} tokens "
        f"(平均 Prompt 占比 {workload['shared_prefix_ratio'] * 100:.1f}%)"
    )
    print(f"{indent}独有 Prompt 长度（实际请求）   : {_format_token_stats(workload['unique_prompt_tokens'])} tokens")


def torch_sync():
    """确保 GPU 计算队列清空后再计时，避免异步执行导致计时不准"""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def run_offline_benchmark(args):
    """使用 vLLM 离线 LLM 类进行精确测量"""
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    total_requests = args.num_prompts if args.num_prompts is not None else args.concurrency
    if total_requests < args.concurrency:
        print(f"[警告] 总请求数 ({total_requests}) 小于并发数 ({args.concurrency})，自动将总请求数调整为并发数 ({args.concurrency})")
        total_requests = args.concurrency

    tokenizer_path = args.tokenizer or args.model
    print(f"[1/4] 加载 tokenizer: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    run_uuid = uuid.uuid4().hex[:8] if args.random_shared_prefix else ""
    print(f"[2/4] 构造测试 prompt (dataset={args.dataset}, 总请求数={total_requests}, 并发数={args.concurrency}) ...")
    batch = build_request_batch(args, tokenizer, total_requests, run_id=run_uuid)
    prompts = batch.prompts
    workload = summarize_dataset_batch(batch, args.dataset)
    shared_len = workload["shared_prefix_tokens"]

    print_workload_summary(workload, indent="      ")

    print(f"[3/4] 初始化 vLLM 引擎 (tensor_parallel_size={args.tp_size}, dtype=bfloat16) ...")
    max_prompt_len = max(batch.prompt_lens)
    max_output_len = max(batch.output_lens)
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp_size,
        max_model_len=max_prompt_len + max_output_len,
        gpu_memory_utilization=args.gpu_mem_util,
        trust_remote_code=True,
        dtype="bfloat16",
    )

    sampling_params = [
        SamplingParams(max_tokens=output_len, temperature=0.0)
        for output_len in batch.output_lens
    ]

    if not args.no_warmup:
        print("[4/4] 基础环境预热 (排除 CUDA Graph 捕获与编译/初始化开销) ...")
        _ = llm.generate(["hello"], SamplingParams(max_tokens=1))
        torch_sync()

        if shared_len > 0:
            print(f"      发送 1 轮共享前缀预热请求 (预热 {shared_len} tokens 共享前缀至 kv cache) ...")
            shared_prompt = tokenizer.decode(tokenizer.encode(prompts[0])[:shared_len])
            _ = llm.generate([shared_prompt], SamplingParams(max_tokens=1))
            torch_sync()
            print("      共享前缀预热完成 (Cache 已填满)。")
    else:
        print("[4/4] 跳过预热 (已传入 --no-warmup) ...")

    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    torch_sync()
    t1 = time.perf_counter()

    elapsed = t1 - t0
    total_prompt_tokens = sum(batch.prompt_lens)
    total_generated_tokens = sum([len(o.outputs[0].token_ids) for o in outputs])
    avg_generated_tokens = total_generated_tokens / total_requests
    all_tokens = total_prompt_tokens + total_generated_tokens

    has_decode = any(output_len > 1 for output_len in batch.output_lens)

    print("\n========== 结果 ==========")
    print(f"并发请求数 (Concurrency)               : {args.concurrency}")
    print(f"总请求数 (Total Requests)             : {total_requests}")
    print_workload_summary(workload)
    print(f"实际平均生成 Token 数                  : {avg_generated_tokens:.1f}")
    print(f"测试总时间 (包含 Prefill + Decode)    : {elapsed:.3f} 秒")
    if not has_decode:
        print(f"Prefill 阶段系统总吞吐量 (Total Prompts / 耗时): {total_prompt_tokens / elapsed:.1f} tokens/秒")
    print(f"系统全流程综合吞吐量 (Tokens / 总时间): {all_tokens / elapsed:.1f} tokens/秒")
    print(f"单请求平均总耗时                      : {elapsed / total_requests:.3f} 秒")
    print("===========================")
    return {
        "scenario": "offline",
        "metrics": {
            "concurrency": args.concurrency,
            "total_requests": total_requests,
            "wall_time": elapsed,
            "total_prompt_tokens": total_prompt_tokens,
            "total_generated_tokens": total_generated_tokens,
            "avg_generated_tokens": avg_generated_tokens,
            "overall_throughput": all_tokens / elapsed,
            "prefill_throughput": total_prompt_tokens / elapsed if not has_decode else None,
            "avg_request_time": elapsed / total_requests,
        },
        "workload": workload,
        "prompt": {
            "tokens": workload["prompt_tokens"],
            "shared_tokens": workload["shared_prefix_tokens"],
            "unique_tokens": workload["unique_prompt_tokens"],
        },
    }


def send_single_api_request(req_id, prompt, url, headers, model, max_tokens, tokenizer=None, ignore_eos=True):
    """发送单个 API 请求并测量 TTFT 与生成耗时 (持续接收数据直到 max_tokens 结束)"""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": ignore_eos,
    }
    t0 = time.perf_counter()
    request_start_timestamp = t0
    first_token_time = None
    last_token_time = None
    estimated_itl_samples = []
    first_token_text = ""
    full_text = ""
    usage_completion_tokens = None
    usage_prompt_tokens = None
    output_token_source = "unavailable"

    try:
        with requests.post(url, json=payload, headers=headers, stream=True, timeout=600) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                decoded = line.decode("utf-8")
                if not decoded.startswith("data: "):
                    continue
                data_str = decoded[len("data: "):]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                # Prefer server-side usage because it includes chat-template/system tokens.
                usage = chunk.get("usage")
                if usage:
                    if usage.get("completion_tokens") is not None:
                        usage_completion_tokens = usage["completion_tokens"]
                    if usage.get("prompt_tokens") is not None:
                        usage_prompt_tokens = usage["prompt_tokens"]

                choices = chunk.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    first_text = delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning")
                    if first_text:
                        now = time.perf_counter()
                        if first_token_time is None:
                            first_token_time = now
                            first_token_text = first_text
                        else:
                            # A streaming chunk may contain multiple model tokens.
                            # Estimate token count locally, then spread this chunk's
                            # arrival interval evenly across those tokens. Repeating
                            # the value makes aggregate averages and percentiles
                            # token-weighted rather than chunk-weighted.
                            chunk_tokens = 1
                            if tokenizer is not None:
                                try:
                                    chunk_tokens = max(
                                        1,
                                        len(tokenizer.encode(
                                            first_text,
                                            add_special_tokens=False,
                                        )),
                                    )
                                except (TypeError, ValueError):
                                    # Some tokenizer-compatible implementations do
                                    # not accept add_special_tokens.
                                    chunk_tokens = max(1, len(tokenizer.encode(first_text)))
                            estimated_itl = (now - last_token_time) / chunk_tokens
                            estimated_itl_samples.extend([estimated_itl] * chunk_tokens)
                        last_token_time = now
                        full_text += first_text
    except Exception as e:
        return {
            "req_id": req_id, "error": str(e), "ttft": None, "tpot": None,
            "request_start_timestamp": request_start_timestamp,
            "first_token_timestamp": None, "last_token_timestamp": None,
            "first_token_text": "", "prompt_tokens": None, "output_tokens": None,
            "output_token_source": "unavailable", "total_time": time.perf_counter() - t0,
            "estimated_itl_samples": [],
        }

    if first_token_time is None:
        return {
            "req_id": req_id, "error": "未收到首个 token", "ttft": None,
            "tpot": None, "request_start_timestamp": request_start_timestamp,
            "first_token_timestamp": None, "last_token_timestamp": None,
            "first_token_text": "", "prompt_tokens": usage_prompt_tokens,
            "output_tokens": None, "output_token_source": "unavailable",
            "total_time": time.perf_counter() - t0, "estimated_itl_samples": [],
        }

    total_time = time.perf_counter() - t0
    ttft = first_token_time - t0

    # Prefer server usage. The local fallback intentionally excludes tokenizer
    # special tokens, matching the chunk-level ITL approximation. Without either
    # source, an SSE chunk count is not a token count and remains unavailable.
    if isinstance(usage_completion_tokens, int) and usage_completion_tokens >= 0:
        actual_output_tokens = usage_completion_tokens
        output_token_source = "server_usage"
    elif tokenizer is not None and full_text:
        try:
            actual_output_tokens = len(tokenizer.encode(full_text, add_special_tokens=False))
        except (TypeError, ValueError):
            actual_output_tokens = len(tokenizer.encode(full_text))
        output_token_source = "local_tokenizer"
    else:
        actual_output_tokens = None

    # TPOT measures token-arrival intervals, not the later [DONE]/HTTP-close
    # delay. A one-token response has no inter-token interval and is therefore
    # explicitly not applicable rather than treated as a zero-latency sample.
    tpot = None
    if (
        actual_output_tokens is not None
        and actual_output_tokens > 1
        and last_token_time is not None
        and last_token_time > first_token_time
    ):
        tpot = (last_token_time - first_token_time) / (actual_output_tokens - 1)

    return {
        "req_id": req_id,
        "error": None,
        "ttft": ttft,
        "tpot": tpot,
        "request_start_timestamp": request_start_timestamp,
        "first_token_timestamp": first_token_time,
        "last_token_timestamp": last_token_time,
        "first_token_text": first_token_text,
        "prompt_tokens": usage_prompt_tokens,
        "output_tokens": actual_output_tokens,
        "output_token_source": output_token_source,
        "total_time": total_time,
        "estimated_itl_samples": estimated_itl_samples,
    }


def _percentiles(values, pcts=(0.5, 0.9, 0.95, 0.99)):
    """Compute linearly interpolated percentiles using positions on [0, n-1]."""
    import math

    if not values:
        return {p: None for p in pcts}
    sorted_values = sorted(values)
    last_index = len(sorted_values) - 1
    result = {}
    for percentile in pcts:
        position = last_index * percentile
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            result[percentile] = sorted_values[lower]
        else:
            fraction = position - lower
            result[percentile] = (
                sorted_values[lower] * (1 - fraction)
                + sorted_values[upper] * fraction
            )
    return result


METRICS_SAMPLE_INTERVAL_SECONDS = 1.0


def _metrics_base_url(api_base):
    """Return the Prometheus endpoint base URL from an OpenAI-compatible API URL."""
    if api_base.endswith("/v1/"):
        api_base = api_base[:-4]
    elif api_base.endswith("/v1"):
        api_base = api_base[:-3]
    return api_base.rstrip("/")


def _parse_prometheus_sample(line):
    """Return a metric name and value from one Prometheus text-format sample."""
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    fields = line.split(None, 1)
    if len(fields) != 2:
        return None
    name_and_labels, values = fields
    metric_name = name_and_labels.split("{", 1)[0]
    value_fields = values.split()
    if not value_fields:
        return None
    try:
        # The first field is always the sample value; a second field, when
        # present, is the optional Prometheus timestamp.
        return metric_name, float(value_fields[0])
    except ValueError:
        return None


def _metric_as_percent(value):
    """Normalize Prometheus ratios (0..1) and already-percent values (0..100)."""
    return value * 100 if 0 <= value <= 1 else value


def _set_max_metric(info, key, value, source=None):
    """Keep the highest labelled engine/worker sample for an instantaneous metric."""
    if value > info.get(key, float("-inf")):
        info[key] = value
        if source:
            info[f"{key}_source"] = source


def query_gpu_metrics(api_base, headers=None):
    """Query a vLLM or SGLang Prometheus endpoint for current KV/cache metrics.

    The API authentication header is forwarded because deployments commonly
    protect /metrics behind the same gateway as /v1/chat/completions.
    """
    if requests is None:
        return {}

    info = {}
    try:
        response = requests.get(
            f"{_metrics_base_url(api_base)}/metrics",
            headers=headers,
            timeout=5,
        )
        if response.status_code != 200:
            return info

        for line in response.text.splitlines():
            sample = _parse_prometheus_sample(line)
            if sample is None:
                continue
            metric_name, value = sample

            # vLLM exposes KV cache usage directly. SGLang's documented
            # sglang:token_usage gauge is the equivalent KV token utilization;
            # do not treat the unrelated/non-standard sglang:utilization as
            # cache usage.
            if metric_name in {
                "gpu_cache_usage_perc",
                "vllm:kv_cache_usage_perc",
                "vllm_kv_cache_usage_perc",
            }:
                _set_max_metric(
                    info,
                    "gpu_cache_usage_pct",
                    _metric_as_percent(value),
                    metric_name,
                )
            elif metric_name in {"cpu_cache_usage_perc", "vllm:cpu_cache_usage_perc"}:
                _set_max_metric(
                    info,
                    "cpu_cache_usage_pct",
                    _metric_as_percent(value),
                    metric_name,
                )
            elif metric_name in {
                "num_requests_running",
                "vllm:num_requests_running",
                "sglang:num_running_reqs",
            }:
                _set_max_metric(info, "running_requests", int(value), metric_name)
            elif metric_name in {
                "num_requests_waiting",
                "vllm:num_requests_waiting",
                "sglang:num_queue_reqs",
            }:
                _set_max_metric(info, "waiting_requests", int(value), metric_name)
            elif metric_name in {"sglang:cache_hit_rate", "sglang_cache_hit_rate"}:
                _set_max_metric(info, "cache_hit_rate", _metric_as_percent(value), metric_name)
            elif metric_name in {"sglang:token_usage", "sglang_token_usage"}:
                _set_max_metric(
                    info,
                    "gpu_cache_usage_pct",
                    _metric_as_percent(value),
                    metric_name,
                )
    except Exception:
        # Server metrics must not fail a benchmark request round.
        pass
    return info


SERVER_RESOURCE_METRIC_KEYS = (
    "gpu_cache_usage_pct",
    "cpu_cache_usage_pct",
    "running_requests",
    "waiting_requests",
    "cache_hit_rate",
)
SERVER_RESOURCE_PERCENTILES = (0.5, 0.9, 0.95, 0.99)


def _active_server_metric_samples(samples):
    """Return snapshots that overlap server-side request activity.

    The metrics sampler starts during a benchmark window, so its first or last
    response can legitimately describe an idle SGLang scheduler. When the
    backend exposes running/queued request gauges, those idle zero snapshots
    must not lower utilization, queue, or cache percentiles. If a backend does
    not expose either gauge, preserve the historical behavior and use all
    successfully fetched snapshots.
    """
    activity_keys = ("running_requests", "waiting_requests")
    snapshots_with_activity_gauges = [
        snapshot
        for snapshot in samples
        if any(isinstance(snapshot.get(key), (int, float)) for key in activity_keys)
    ]
    if not snapshots_with_activity_gauges:
        return list(samples), False
    return [
        snapshot
        for snapshot in snapshots_with_activity_gauges
        if any(snapshot.get(key, 0) > 0 for key in activity_keys)
    ], True


def _peak_server_metrics(samples):
    """Summarize resource snapshots collected during one benchmark window.

    Each snapshot already represents the maximum value across labelled
    workers/engines. When a backend supplies running/queued gauges, only
    snapshots with active server-side work contribute to min/avg/max and
    percentiles; idle zero snapshots are retained only in metadata.
    """
    active_samples, activity_gauges_available = _active_server_metric_samples(samples)
    summary = {
        "sample_count": len(active_samples),
        "raw_sample_count": len(samples),
        "inactive_sample_count": len(samples) - len(active_samples),
        "activity_gauges_available": activity_gauges_available,
        "metrics": {},
    }
    for key in SERVER_RESOURCE_METRIC_KEYS:
        values = [
            snapshot[key]
            for snapshot in active_samples
            if isinstance(snapshot.get(key), (int, float))
        ]
        if not values:
            continue

        percentiles = _percentiles(values, SERVER_RESOURCE_PERCENTILES)
        maximum = max(values)
        max_snapshot = next(
            snapshot
            for snapshot in active_samples
            if snapshot.get(key) == maximum
        )
        stats = {
            "sample_count": len(values),
            "min": min(values),
            "avg": sum(values) / len(values),
            "max": maximum,
            "p50": percentiles[0.5],
            "p90": percentiles[0.9],
            "p95": percentiles[0.95],
            "p99": percentiles[0.99],
        }
        source = max_snapshot.get(f"{key}_source")
        if source:
            stats["max_source"] = source

        summary["metrics"][key] = stats
        # Keep the historical peak_* fields for consumers of existing reports.
        summary[f"peak_{key}"] = maximum
        if source:
            summary[f"peak_{key}_source"] = source
    return summary


def _format_server_metric_value(key, value):
    if key in {"gpu_cache_usage_pct", "cpu_cache_usage_pct", "cache_hit_rate"}:
        return f"{value:.1f}%"
    return f"{value:.0f}"


def _print_server_metrics(metrics):
    """Print active test-window resource statistics without idle snapshots."""
    if not metrics or not metrics.get("raw_sample_count"):
        return

    print("\n  ── 服务端资源 (仅测试过程采样) ──")
    print(f"    原始运行期采样次数        : {metrics['raw_sample_count']}")
    print(f"    有效活跃采样次数          : {metrics['sample_count']}")
    if metrics.get("activity_gauges_available") and not metrics["sample_count"]:
        print("    未观测到服务端运行/排队请求；不展示可能误导的空闲 0 值统计")
        return
    if metrics.get("inactive_sample_count"):
        print(f"    已排除空闲采样次数        : {metrics['inactive_sample_count']}")
    labels = {
        "gpu_cache_usage_pct": "GPU KV Cache",
        "cpu_cache_usage_pct": "CPU KV Cache",
        "running_requests": "运行中请求",
        "waiting_requests": "排队请求",
        "cache_hit_rate": "Cache 命中率",
    }
    for key in SERVER_RESOURCE_METRIC_KEYS:
        stats = metrics.get("metrics", {}).get(key)
        if not stats:
            continue
        value_text = " / ".join(
            _format_server_metric_value(key, stats[name])
            for name in ("min", "avg", "max")
        )
        print(f"    {labels[key]:<18s} min / avg / max : {value_text}")
        percentile_text = " / ".join(
            _format_server_metric_value(key, stats[f"p{int(percentile * 100)}"])
            for percentile in SERVER_RESOURCE_PERCENTILES
        )
        print(f"    {'':<18s} P50 / P90 / P95 / P99 : {percentile_text}")
        if stats.get("max_source"):
            print(f"    {'':<18s} 最大值来源        : {stats['max_source']}")


def _is_finite_number(value):
    """Return whether value is a non-boolean finite numeric measurement."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _format_duration(value, *, milliseconds=False):
    """Format an optional latency value for compact per-round summaries."""
    if not _is_finite_number(value):
        return "n/a"
    if milliseconds:
        return f"{value * 1000:.2f}ms"
    return f"{value:.3f}s"


def _interval_union_duration(intervals):
    """Return the elapsed duration covered by the union of valid time intervals."""
    valid_intervals = sorted(
        (start, end)
        for start, end in intervals
        if _is_finite_number(start) and _is_finite_number(end) and end > start
    )
    if not valid_intervals:
        return None

    duration = 0.0
    current_start, current_end = valid_intervals[0]
    for start, end in valid_intervals[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            duration += current_end - current_start
            current_start, current_end = start, end
    return duration + current_end - current_start


def _request_meets_slo(result, slo_ttft, slo_tpot):
    """Return whether one completed request satisfies the configured SLOs."""
    ttft = result.get("ttft")
    if not _is_finite_number(ttft) or ttft > slo_ttft:
        return False
    output_tokens = result.get("output_tokens")
    if not isinstance(output_tokens, int) or isinstance(output_tokens, bool) or output_tokens < 0:
        # A decode SLO cannot be verified without a trustworthy output count.
        return False
    if output_tokens <= 1:
        return True
    tpot = result.get("tpot")
    return _is_finite_number(tpot) and tpot <= slo_tpot


def _aggregate_api_round_metrics(
    results, prompt_lens_list, max_tokens_list, concurrency, wall_time,
    server_metrics, slo_ttft, slo_tpot,
):
    """Aggregate request records using explicit, consistent result populations."""
    total_requests = len(prompt_lens_list)
    results = sorted(results, key=lambda result: result.get("req_id", -1))
    successful_results = [
        result for result in results if _is_finite_number(result.get("ttft"))
    ]
    successful_count = len(successful_results)
    failed_count = total_requests - successful_count

    successful_ttfts = sorted(result["ttft"] for result in successful_results)
    ttft_pcts = _percentiles(successful_ttfts)
    successful_totals = [
        result["total_time"]
        for result in successful_results
        if _is_finite_number(result.get("total_time"))
    ]
    e2e_pcts = _percentiles(successful_totals)

    prompt_counts = []
    server_prompt_usage_count = 0
    for result in successful_results:
        server_count = result.get("prompt_tokens")
        if isinstance(server_count, int) and not isinstance(server_count, bool) and server_count > 0:
            prompt_counts.append(server_count)
            server_prompt_usage_count += 1
        else:
            prompt_counts.append(prompt_lens_list[result["req_id"]])
    total_prompt_tokens = sum(prompt_counts)

    output_counted_results = [
        result for result in successful_results
        if isinstance(result.get("output_tokens"), int)
        and not isinstance(result["output_tokens"], bool)
        and result["output_tokens"] >= 0
    ]
    # Empty success populations must not make output measurement vacuously valid.
    output_token_count_complete = (
        successful_count > 0 and len(output_counted_results) == successful_count
    )
    counted_generated_tokens = sum(
        result["output_tokens"] for result in output_counted_results
    )
    total_generated_tokens = (
        counted_generated_tokens if output_token_count_complete else None
    )
    output_token_sources = {}
    for result in output_counted_results:
        source = result.get("output_token_source", "unavailable")
        output_token_sources[source] = output_token_sources.get(source, 0) + 1

    tpot_samples = [
        result["tpot"]
        for result in output_counted_results
        if _is_finite_number(result.get("tpot")) and result["output_tokens"] > 1
    ]
    tpot_pcts = _percentiles(tpot_samples)
    all_estimated_itl_samples = [
        sample
        for result in successful_results
        for sample in result.get("estimated_itl_samples", [])
        if _is_finite_number(sample)
    ]
    estimated_itl_pcts = _percentiles(all_estimated_itl_samples)

    prompt_workload_throughput = (
        total_prompt_tokens / wall_time
        if successful_count and _is_finite_number(wall_time) and wall_time > 0
        else None
    )
    prefill_intervals = [
        (result.get("request_start_timestamp"), result.get("first_token_timestamp"))
        for result in successful_results
    ]
    prefill_duration = _interval_union_duration(prefill_intervals)
    prefill_throughput = (
        total_prompt_tokens / prefill_duration
        if _is_finite_number(prefill_duration) and prefill_duration > 0
        else None
    )

    decode_results = [
        result for result in output_counted_results if result["output_tokens"] > 1
    ]
    decode_token_count_complete = (
        output_token_count_complete
        and all(
            _is_finite_number(result.get("first_token_timestamp"))
            and _is_finite_number(result.get("last_token_timestamp"))
            and result["last_token_timestamp"] > result["first_token_timestamp"]
            for result in decode_results
        )
    )
    total_decode_tokens = sum(result["output_tokens"] - 1 for result in decode_results)
    decode_duration = _interval_union_duration([
        (result.get("first_token_timestamp"), result.get("last_token_timestamp"))
        for result in decode_results
    ])
    decode_throughput = (
        total_decode_tokens / decode_duration
        if (
            total_decode_tokens > 0
            and decode_token_count_complete
            and _is_finite_number(decode_duration)
            and decode_duration > 0
        )
        else None
    )
    overall_throughput = (
        (total_prompt_tokens + total_generated_tokens) / wall_time
        if output_token_count_complete and _is_finite_number(wall_time) and wall_time > 0
        else None
    )
    goodput_count = sum(
        _request_meets_slo(result, slo_ttft, slo_tpot)
        for result in successful_results
    )
    timestamped_results = [
        result for result in successful_results
        if _is_finite_number(result.get("first_token_timestamp"))
    ]
    earliest_token_result = min(
        timestamped_results,
        key=lambda result: result["first_token_timestamp"],
        default=None,
    )

    return {
        "concurrency": concurrency,
        "total_requests": total_requests,
        "successful": successful_count,
        "failed": failed_count,
        "failure_rate": failed_count / total_requests if total_requests else 0.0,
        "wall_time": wall_time,
        "avg_ttft": sum(successful_ttfts) / successful_count if successful_count else None,
        "min_ttft": successful_ttfts[0] if successful_ttfts else None,
        "max_ttft": successful_ttfts[-1] if successful_ttfts else None,
        "p50_ttft": ttft_pcts[0.5], "p90_ttft": ttft_pcts[0.9],
        "p95_ttft": ttft_pcts[0.95], "p99_ttft": ttft_pcts[0.99],
        "avg_total_time": sum(successful_totals) / len(successful_totals) if successful_totals else None,
        "p50_e2e": e2e_pcts[0.5], "p90_e2e": e2e_pcts[0.9], "p99_e2e": e2e_pcts[0.99],
        "prompt_throughput": prompt_workload_throughput,
        "prefill_throughput": prefill_throughput,
        "prefill_duration": prefill_duration if prefill_throughput is not None else None,
        "decode_throughput": decode_throughput,
        "decode_duration": decode_duration if decode_throughput is not None else None,
        "total_decode_tokens": (
            total_decode_tokens if output_token_count_complete else None
        ),
        "decode_token_count_complete": decode_token_count_complete,
        "overall_throughput": overall_throughput,
        "qps": successful_count / wall_time if _is_finite_number(wall_time) and wall_time > 0 else 0.0,
        "avg_tpot": sum(tpot_samples) / len(tpot_samples) if tpot_samples else None,
        "p50_tpot": tpot_pcts[0.5], "p90_tpot": tpot_pcts[0.9],
        "p95_tpot": tpot_pcts[0.95], "p99_tpot": tpot_pcts[0.99],
        "avg_estimated_itl": (
            sum(all_estimated_itl_samples) / len(all_estimated_itl_samples)
            if all_estimated_itl_samples else None
        ),
        "p50_estimated_itl": estimated_itl_pcts[0.5],
        "p90_estimated_itl": estimated_itl_pcts[0.9],
        "p99_estimated_itl": estimated_itl_pcts[0.99],
        "goodput_pct": goodput_count / total_requests * 100 if total_requests else 0.0,
        "slo_passed": goodput_count,
        "goodput_qps": goodput_count / wall_time if _is_finite_number(wall_time) and wall_time > 0 else 0.0,
        "slo_ttft": slo_ttft,
        "slo_tpot": slo_tpot,
        "total_prompt_tokens": total_prompt_tokens,
        "server_prompt_usage_count": server_prompt_usage_count,
        "prompt_token_source": (
            "server_usage" if successful_count and server_prompt_usage_count == successful_count
            else "mixed_server_and_local" if server_prompt_usage_count
            else "local_estimate"
        ),
        "total_generated_tokens": total_generated_tokens,
        "output_token_counted_requests": len(output_counted_results),
        "output_token_count_complete": output_token_count_complete,
        "output_token_sources": output_token_sources,
        "avg_generated_tokens": (
            total_generated_tokens / successful_count
            if output_token_count_complete else None
        ),
        "avg_prompt_len": total_prompt_tokens / successful_count if successful_count else None,
        "server_metrics": server_metrics,
        "server_metrics_peak": server_metrics,
        "earliest_token_request_id": (
            earliest_token_result["req_id"] if earliest_token_result else None
        ),
        "results": results,
    }


def run_api_benchmark_round(prompts, prompt_lens, url, headers, model, max_tokens, concurrency, tokenizer=None, ignore_eos=True, slo_ttft=5.0, slo_tpot=0.1):
    """
    Execute a single benchmark round: send concurrent requests, collect results, compute metrics.

    Args:
        prompts: list of prompt strings
        prompt_lens: int (uniform) or list of ints (per-request prompt token counts)
        max_tokens: int (uniform) or list of ints (per-request output token limits)

    Returns a dict with all computed metrics, or None if all requests failed.
    """
    total_requests = len(prompts)
    # Normalize to per-request lists
    if isinstance(prompt_lens, int):
        prompt_lens_list = [prompt_lens] * total_requests
    else:
        prompt_lens_list = list(prompt_lens)
    if isinstance(max_tokens, int):
        max_tokens_list = [max_tokens] * total_requests
    else:
        max_tokens_list = list(max_tokens)

    wall_t0 = time.perf_counter()
    results = []
    completed = [0]  # use list for mutability in closure
    succeeded = [0]
    failed_cnt = [0]
    last_ttft = [None]
    progress_lock = threading.Lock()

    metrics_api_base = (
        url[:-len("/chat/completions")]
        if url.endswith("/chat/completions")
        else url
    )
    server_metric_samples = []
    server_metrics_lock = threading.Lock()
    stop_metrics_sampling = threading.Event()

    def capture_server_metrics():
        snapshot = query_gpu_metrics(metrics_api_base, headers=headers)
        sample_completed_at = time.perf_counter()
        if snapshot:
            # Only samples whose HTTP response has already arrived before
            # wall_t1 are included in the run-time peak.
            snapshot["elapsed_seconds"] = sample_completed_at - wall_t0
            with server_metrics_lock:
                server_metric_samples.append(snapshot)

    def sample_server_metrics():
        capture_server_metrics()
        while not stop_metrics_sampling.wait(METRICS_SAMPLE_INTERVAL_SECONDS):
            capture_server_metrics()

    metrics_sampler = threading.Thread(
        target=sample_server_metrics,
        name="benchmark-server-metrics",
        daemon=True,
    )
    metrics_sampler.start()

    def _on_complete(future):
        try:
            r = future.result()
        except BaseException as exc:
            # A worker normally converts request errors into result records, but
            # keep unexpected executor failures from silently dropping requests.
            r = {
                "req_id": getattr(future, "benchmark_req_id", None),
                "error": f"{type(exc).__name__}: {exc}",
                "ttft": None,
                "tpot": None,
                "request_start_timestamp": None,
                "first_token_timestamp": None,
                "last_token_timestamp": None,
                "first_token_text": "",
                "prompt_tokens": None,
                "output_tokens": None,
                "output_token_source": "unavailable",
                "total_time": None,
                "estimated_itl_samples": [],
            }
        with progress_lock:
            results.append(r)
            completed[0] += 1
            request_failed = not _is_finite_number(r.get("ttft"))
            if not request_failed:
                succeeded[0] += 1
                last_ttft[0] = r["ttft"]
            else:
                failed_cnt[0] += 1
                reason = r.get("error") or "未收到有效的首个 token"
                print(
                    f"\n  请求失败: req_id={r.get('req_id', 'unknown')}  原因: {reason}",
                    flush=True,
                )
            elapsed = time.perf_counter() - wall_t0
            ttft_str = f"  最新TTFT: {last_ttft[0]:.3f}s" if last_ttft[0] is not None else ""
            fail_str = f"  失败: {failed_cnt[0]}" if failed_cnt[0] > 0 else ""
            print(f"\r  进度: [{completed[0]:>{len(str(total_requests))}d}/{total_requests}]"
                  f"  耗时: {elapsed:.1f}s{ttft_str}{fail_str}    ", end="", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = []
        for i in range(total_requests):
            f = executor.submit(send_single_api_request, i, prompts[i], url, headers, model, max_tokens_list[i], tokenizer, ignore_eos)
            f.benchmark_req_id = i
            f.add_done_callback(_on_complete)
            futures.append(f)
        concurrent.futures.wait(futures)

    wall_t1 = time.perf_counter()
    stop_metrics_sampling.set()
    metrics_sampler.join(timeout=6)
    with server_metrics_lock:
        runtime_metric_samples = [
            sample
            for sample in server_metric_samples
            if sample.get("elapsed_seconds", float("inf")) <= wall_t1 - wall_t0
        ]
    server_metrics = _peak_server_metrics(runtime_metric_samples)

    wall_time = wall_t1 - wall_t0
    print(f"\r  进度: [{total_requests}/{total_requests}]  完成!  总耗时: {wall_time:.3f}s"
          f"  成功: {succeeded[0]}  失败: {failed_cnt[0]}        ")


    return _aggregate_api_round_metrics(
        results,
        prompt_lens_list,
        max_tokens_list,
        concurrency,
        wall_time,
        server_metrics,
        slo_ttft,
        slo_tpot,
    )


def print_benchmark_metrics(metrics, workload):
    """Pretty-print API metrics and the generated workload actually sent."""
    m = metrics  # shorthand

    print("\n========== 测试结果 ==========")
    # ── Basic info ──
    print(f"并发请求数                  : {m['concurrency']}")
    print(f"总请求数                    : {m['total_requests']}  (成功: {m['successful']}, 失败: {m['failed']})")
    print_workload_summary(workload)
    print(
        f"请求输出上限 / 实际平均生成输出: "
        f"{_format_token_stats(workload['requested_output_tokens'])} / "
        f"{m['avg_generated_tokens']:.1f} tokens"
        if m.get("avg_generated_tokens") is not None
        else "请求输出上限 / 实际平均生成输出: "
             f"{_format_token_stats(workload['requested_output_tokens'])} / 不可验证"
    )
    print(f"测试总时间                  : {m['wall_time']:.3f} 秒")
    if not m["successful"]:
        print("\n  所有请求均未收到首个 token；TTFT/TPOT/吞吐量均不可计算。")
        print(f"  失败率                    : {m['failure_rate'] * 100:.1f}%")
        _print_server_metrics(m.get("server_metrics", m.get("server_metrics_peak", {})))
        print("=============================")
        return

    # ── TTFT ──
    print(f"\n  ── TTFT (首 Token 延迟) ──")
    print(f"    平均 / 最小 / 最大      : {m['avg_ttft']:.3f} / {m['min_ttft']:.3f} / {m['max_ttft']:.3f} 秒")
    if m['successful'] > 1:
        print(f"    P50 / P90 / P99         : {m['p50_ttft']:.3f} / {m['p90_ttft']:.3f} / {m['p99_ttft']:.3f} 秒")

    # ── E2E Latency ──
    print(f"\n  ── E2E 端到端延迟 ──")
    if m.get('avg_total_time') is not None:
        print(f"    平均                    : {m['avg_total_time']:.3f} 秒")
    else:
        print("    平均                    : 不可验证")
    if m.get('p50_e2e') is not None:
        print(f"    P50 / P90 / P99         : {m['p50_e2e']:.3f} / {m['p90_e2e']:.3f} / {m['p99_e2e']:.3f} 秒")

    # ── TPOT ──
    if m['avg_tpot'] is not None:
        print(f"\n  ── TPOT (每输出 Token 耗时) ──")
        print(f"    平均                    : {m['avg_tpot']*1000:.1f} ms")
        if m.get('p50_tpot') is not None:
            print(f"    P50 / P90 / P99         : {m['p50_tpot']*1000:.1f} / {m['p90_tpot']*1000:.1f} / {m['p99_tpot']*1000:.1f} ms")

    # ── Estimated ITL ──
    if m['avg_estimated_itl'] is not None:
        print(f"\n  ── 预估 ITL (基于流式 Chunk 本地分词) ──")
        print(f"    平均                    : {m['avg_estimated_itl']*1000:.1f} ms")
        if m.get('p50_estimated_itl') is not None:
            print(f"    P50 / P90 / P99         : {m['p50_estimated_itl']*1000:.1f} / {m['p90_estimated_itl']*1000:.1f} / {m['p99_estimated_itl']*1000:.1f} ms")

    # ── Throughput ──
    print(f"\n  ── 吞吐量（成功且 token 数可验证的请求） ──")
    if m.get("prompt_throughput") is not None:
        print(f"    Prompt 全流程工作负载率  : {m['prompt_throughput']:.1f} tokens/s  (prompt tokens / 全测试窗口)")
    if m.get("prefill_throughput") is not None:
        print(f"    Prefill 阶段吞吐量       : {m['prefill_throughput']:.1f} tokens/s  (prompt tokens / 活动窗口 {m['prefill_duration']:.3f}s)")
    else:
        print("    Prefill 阶段吞吐量       : 不可验证（缺少请求发送或首 token 时间）")
    if m.get("decode_throughput") is not None:
        print(f"    Decode 阶段吞吐量        : {m['decode_throughput']:.1f} tokens/s  (输出 token（不含首 token）/ 活动窗口 {m['decode_duration']:.3f}s)")
    elif m.get("total_decode_tokens"):
        print("    Decode 阶段吞吐量        : 不可验证（缺少独立内容 token 到达时间）")
    if m.get("overall_throughput") is not None:
        print(f"    整体交付吞吐量           : {m['overall_throughput']:.1f} tokens/s  (prompt + output / 全测试窗口)")
    else:
        print("    整体交付吞吐量           : 不可验证（存在无法计数的输出 token）")
    print(f"    QPS                     : {m['qps']:.2f} req/s")

    # ── Goodput ──
    if m.get('goodput_pct') is not None:
        print(f"\n  ── Goodput (SLO 达标率) ──")
        print(f"    SLO: TTFT ≤ {m['slo_ttft']:.3g}s AND TPOT ≤ {m['slo_tpot'] * 1000:.3g}ms")
        print(f"    达标率                  : {m['goodput_pct']:.1f}%  ({m['slo_passed']}/{m['total_requests']} 请求)")
        print(f"    Goodput QPS             : {m['goodput_qps']:.2f} req/s")

    # ── GPU / KV Cache Utilization ──
    _print_server_metrics(m.get("server_metrics", m.get("server_metrics_peak", {})))

    first_result = min(
        (result for result in m["results"] if result.get("first_token_timestamp") is not None),
        key=lambda result: result["first_token_timestamp"],
        default={},
    )
    print(f"\n最早首个 token 样例          : {first_result.get('first_token_text', '')!r}")
    print("=============================")


def run_api_benchmark(args):
    """
    通过 OpenAI 兼容 API 测量并发 TTFT (Time To First Token)。
    需先用 vllm serve 启动服务。
    """
    from transformers import AutoTokenizer

    total_requests = args.num_prompts if args.num_prompts is not None else args.concurrency
    if total_requests < args.concurrency:
        print(f"[警告] 总请求数 ({total_requests}) 小于并发数 ({args.concurrency})，自动将总请求数调整为并发数 ({args.concurrency})")
        total_requests = args.concurrency

    tokenizer_path = args.tokenizer or args.model
    print(f"[1/4] 加载 tokenizer: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    run_uuid = uuid.uuid4().hex[:8] if args.random_shared_prefix else ""
    print(f"[2/4] 构造测试 prompt (dataset={args.dataset}, 总请求数={total_requests}, 并发数={args.concurrency}) ...")
    batch = build_request_batch(args, tokenizer, total_requests, run_id=run_uuid)
    prompts = batch.prompts
    workload = summarize_dataset_batch(batch, args.dataset)
    shared_len = workload["shared_prefix_tokens"]

    print_workload_summary(workload, indent="      ")

    url = f"{args.api_base}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    if not args.no_warmup:
        print("[3/4] 基础环境预热 (排除 CUDA Graph 捕获与编译/初始化开销) ...")
        warmup_payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": "你好"}],
            "max_tokens": 10,
            "temperature": 0,
            "stream": False,
        }
        try:
            w_t0 = time.perf_counter()
            w_resp = requests.post(url, json=warmup_payload, headers=headers, timeout=120)
            w_resp.raise_for_status()
            w_elapsed = time.perf_counter() - w_t0
            print(f"      基础环境预热完成，耗时: {w_elapsed:.3f} 秒")
        except Exception as e:
            print(f"      基础环境预热警告 (仍继续测试): {e}")

        if shared_len > 0:
            warmup_rounds = args.warmup_rounds
            print(f"      发送 {warmup_rounds} 轮共享前缀预热请求 (预热 {shared_len} tokens 共享前缀至 Cache) ...")
            shared_prompt_text = tokenizer.decode(tokenizer.encode(prompts[0])[:shared_len])
            sp_payload = {
                "model": args.model,
                "messages": [{"role": "user", "content": shared_prompt_text}],
                "max_tokens": 1,
                "temperature": 0,
                "stream": False,
            }
            for r in range(warmup_rounds):
                try:
                    sp_t0 = time.perf_counter()
                    sp_resp = requests.post(url, json=sp_payload, headers=headers, timeout=600)
                    sp_resp.raise_for_status()
                    sp_elapsed = time.perf_counter() - sp_t0
                    print(f"      第 {r+1}/{warmup_rounds} 轮共享前缀预热完成，耗时: {sp_elapsed:.3f} 秒")
                except Exception as e:
                    print(f"      第 {r+1}/{warmup_rounds} 轮共享前缀预热警告 (仍继续测试): {e}")
            print(f"      Prefix KV Cache 预热完毕 ({warmup_rounds} 轮)")
    else:
        print("[3/4] 跳过预热请求 (已传入 --no-warmup) ...")

    print(
        f"[4/4] 发送 {total_requests} 个长上下文流式请求 (并发度={args.concurrency}) "
        f"并测量 TTFT (请求输出上限={_format_token_stats(workload['requested_output_tokens'])}) ..."
    )
    nsys_start()
    metrics = run_api_benchmark_round(
        prompts, batch.prompt_lens, url, headers, args.model, batch.output_lens,
        args.concurrency, tokenizer, args.ignore_eos,
        slo_ttft=args.slo_ttft, slo_tpot=args.slo_tpot
    )
    nsys_stop()
    if not metrics["successful"]:
        print("所有请求均未收到生成内容，请检查服务状态与错误日志")
        print_benchmark_metrics(metrics, workload)
        return {
            "scenario": "single",
            "metrics": metrics,
            "workload": workload,
            "prompt": {
                "tokens": workload["prompt_tokens"],
                "shared_tokens": workload["shared_prefix_tokens"],
                "unique_tokens": workload["unique_prompt_tokens"],
            },
        }

    print_benchmark_metrics(metrics, workload)
    return {
        "scenario": "single",
        "metrics": metrics,
        "workload": workload,
        "prompt": {
            "tokens": workload["prompt_tokens"],
            "shared_tokens": workload["shared_prefix_tokens"],
            "unique_tokens": workload["unique_prompt_tokens"],
        },
    }


# ── Mixed Workload Benchmark ──────────────────────────────────────────

def parse_workload_mix(mix_str: str):
    """
    Parse a workload mix specification string.
    Format: 'input_tokens:output_tokens:weight,input_tokens:output_tokens:weight,...'
    Example: '128:4096:0.3,8192:1024:0.5,131072:1:0.2'

    Returns list of dicts: [{"input": int, "output": int, "weight": float}, ...]
    """
    entries = []
    for part in mix_str.split(","):
        fields = part.strip().split(":")
        if len(fields) != 3:
            raise ValueError(f"Invalid workload mix entry '{part}'. Expected format: input_tokens:output_tokens:weight")
        input_tokens = int(fields[0])
        output_tokens = int(fields[1])
        weight = float(fields[2])
        if input_tokens < 1 or output_tokens < 1 or weight <= 0:
            raise ValueError(
                f"Invalid workload mix entry '{part}': input/output/weight must all be positive"
            )
        entries.append({
            "input": input_tokens,
            "output": output_tokens,
            "weight": weight,
        })
    if not entries:
        raise ValueError("Workload mix must contain at least one entry")
    # Normalize weights
    total_weight = sum(e["weight"] for e in entries)
    for e in entries:
        e["weight"] /= total_weight
    return entries


def run_mixed_benchmark(args):
    """
    Run a mixed workload benchmark with varying context lengths and output lengths.
    Each request is randomly assigned a workload type based on the mix weights.
    """
    from transformers import AutoTokenizer

    mix_str = getattr(args, "workload_mix", None)
    if not mix_str:
        print("ERROR: --workload-mix is required for mixed-workload preset")
        print("       Format: 'input:output:weight,input:output:weight,...'")
        print("       Example: --workload-mix '128:4096:0.3,8192:1024:0.5,131072:1:0.2'")
        return None

    workload_entries = parse_workload_mix(mix_str)
    total_requests = args.num_prompts if args.num_prompts is not None else args.concurrency
    if total_requests < args.concurrency:
        total_requests = args.concurrency

    tokenizer_path = args.tokenizer or args.model
    print(f"[1/4] 加载 tokenizer: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    # Print workload distribution
    print(f"[2/4] 构造混合负载 (总请求数={total_requests}, 并发数={args.concurrency})")
    print(f"      负载分布:")
    for i, e in enumerate(workload_entries):
        count = max(1, round(e["weight"] * total_requests))
        print(f"        类型 {i+1}: input={e['input']:>7d}, output={e['output']:>5d}, 比例={e['weight']*100:.0f}% (~{count} 请求)")

    # Assign each request a workload type based on weights
    request_types = []
    remaining = total_requests
    for i, e in enumerate(workload_entries):
        if i == len(workload_entries) - 1:
            count = remaining  # last type gets the remainder
        else:
            count = max(1, round(e["weight"] * total_requests))
            count = min(count, remaining)
        request_types.extend([i] * count)
        remaining -= count
    random.shuffle(request_types)

    # Generate prompts with varying context lengths
    print(f"      构造 {total_requests} 个混合长度 prompt ...")
    run_uuid = uuid.uuid4().hex[:8]
    dataset_instance = create_dataset(
        args.dataset,
        random_seed=args.random_seed if args.random_seed is not None else args.seed,
    )
    batches = [
        build_request_batch(
            args,
            tokenizer,
            1,
            input_len=workload_entries[type_idx]["input"],
            output_len=workload_entries[type_idx]["output"],
            share_prefix=False,
            run_id=run_uuid,
            dataset=dataset_instance,
        )
        for type_idx in request_types
    ]
    prompts = [batch.prompts[0] for batch in batches]
    prompt_lens = [batch.prompt_lens[0] for batch in batches]
    max_tokens_list = [batch.output_lens[0] for batch in batches]
    # These one-request batches intentionally do not imply a shared prefix
    # across the full mixed request set.
    workload = summarize_workload_lengths(args.dataset, prompt_lens, max_tokens_list)
    type_workloads = [
        summarize_workload_lengths(
            args.dataset,
            [batch.prompt_lens[0] for batch, value in zip(batches, request_types) if value == index],
            [batch.output_lens[0] for batch, value in zip(batches, request_types) if value == index],
        )
        for index in range(len(workload_entries))
    ]

    print_workload_summary(workload, indent="      ")

    url = f"{args.api_base}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    # Warmup
    if not args.no_warmup:
        print("[3/4] 预热服务 ...")
        warmup_payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": "你好"}],
            "max_tokens": 10, "temperature": 0, "stream": False,
        }
        try:
            requests.post(url, json=warmup_payload, headers=headers, timeout=120).raise_for_status()
            print("      预热完成")
        except Exception as e:
            print(f"      预热警告: {e}")
    else:
        print("[3/4] 跳过预热")

    print(f"[4/4] 发送 {total_requests} 个混合负载请求 (并发度={args.concurrency}) ...")
    nsys_start()
    metrics = run_api_benchmark_round(
        prompts, prompt_lens, url, headers, args.model, max_tokens_list,
        args.concurrency, tokenizer, args.ignore_eos,
        slo_ttft=args.slo_ttft, slo_tpot=args.slo_tpot
    )
    nsys_stop()

    if not metrics["successful"]:
        print("所有请求均未收到生成内容，请检查服务状态与错误日志")
        print_benchmark_metrics(metrics, workload)
        return {
            "scenario": "mixed-workload",
            "metrics": metrics,
            "workload": {
                "summary": workload,
                "types": [
                    {
                        "configured_input_tokens": entry["input"],
                        "configured_output_tokens": entry["output"],
                        "weight": entry["weight"],
                        "runtime": type_workloads[index],
                    }
                    for index, entry in enumerate(workload_entries)
                ],
            },
        }

    # Print results with workload distribution info
    m = metrics
    print("\n========== 混合负载测试结果 ==========")
    print(f"并发请求数                  : {m['concurrency']}")
    print(f"总请求数                    : {m['total_requests']}  (成功: {m['successful']}, 失败: {m['failed']})")
    print()
    print("  负载分布（配置值 → 实际请求）:")
    for i, e in enumerate(workload_entries):
        type_workload = type_workloads[i]
        print(
            f"    类型 {i + 1}: 配置 input/output={e['input']}/{e['output']} × "
            f"{type_workload['request_count']} 请求；实际 Prompt/输出上限="
            f"{_format_token_stats(type_workload['prompt_tokens'])}/"
            f"{_format_token_stats(type_workload['requested_output_tokens'])} tokens"
        )
    print()
    print_workload_summary(workload)
    if m.get("avg_generated_tokens") is None:
        print("实际平均生成输出              : 不可验证（存在无法计数的输出 token）")
    else:
        print(f"实际平均生成输出              : {m['avg_generated_tokens']:.1f} tokens")
    print(f"测试总时间                  : {m['wall_time']:.3f} 秒")

    # ── TTFT ──
    print(f"\n  ── TTFT (首 Token 延迟) ──")
    print(f"    平均 / 最小 / 最大      : {m['avg_ttft']:.3f} / {m['min_ttft']:.3f} / {m['max_ttft']:.3f} 秒")
    if m['successful'] > 1:
        print(f"    P50 / P90 / P99         : {m['p50_ttft']:.3f} / {m['p90_ttft']:.3f} / {m['p99_ttft']:.3f} 秒")

    # ── E2E ──
    print(f"\n  ── E2E 端到端延迟 ──")
    if m.get('avg_total_time') is not None:
        print(f"    平均                    : {m['avg_total_time']:.3f} 秒")
    else:
        print("    平均                    : 不可验证")
    if m.get('p50_e2e') is not None:
        print(f"    P50 / P90 / P99         : {m['p50_e2e']:.3f} / {m['p90_e2e']:.3f} / {m['p99_e2e']:.3f} 秒")

    # ── TPOT ──
    if m['avg_tpot'] is not None:
        print(f"\n  ── TPOT (每输出 Token 耗时) ──")
        print(f"    平均                    : {m['avg_tpot']*1000:.1f} ms")
        if m.get('p50_tpot') is not None:
            print(f"    P50 / P90 / P99         : {m['p50_tpot']*1000:.1f} / {m['p90_tpot']*1000:.1f} / {m['p99_tpot']*1000:.1f} ms")

    # ── Estimated ITL ──
    if m['avg_estimated_itl'] is not None:
        print(f"\n  ── 预估 ITL (基于流式 Chunk 本地分词) ──")
        print(f"    平均                    : {m['avg_estimated_itl']*1000:.1f} ms")
        if m.get('p50_estimated_itl') is not None:
            print(f"    P50 / P90 / P99         : {m['p50_estimated_itl']*1000:.1f} / {m['p90_estimated_itl']*1000:.1f} / {m['p99_estimated_itl']*1000:.1f} ms")

    # ── Throughput ──
    print(f"\n  ── 吞吐量（成功且 token 数可验证的请求） ──")
    if m.get("prompt_throughput") is not None:
        print(f"    Prompt 全流程工作负载率  : {m['prompt_throughput']:.1f} tokens/s  (prompt tokens / 全测试窗口)")
    if m.get("prefill_throughput") is not None:
        print(f"    Prefill 阶段吞吐量       : {m['prefill_throughput']:.1f} tokens/s  (prompt tokens / 活动窗口 {m['prefill_duration']:.3f}s)")
    else:
        print("    Prefill 阶段吞吐量       : 不可验证（缺少请求发送或首 token 时间）")
    if m.get("decode_throughput") is not None:
        print(f"    Decode 阶段吞吐量        : {m['decode_throughput']:.1f} tokens/s  (输出 token（不含首 token）/ 活动窗口 {m['decode_duration']:.3f}s)")
    elif m.get("total_decode_tokens"):
        print("    Decode 阶段吞吐量        : 不可验证（缺少独立内容 token 到达时间）")
    if m.get("overall_throughput") is not None:
        print(f"    整体交付吞吐量           : {m['overall_throughput']:.1f} tokens/s")
    else:
        print("    整体交付吞吐量           : 不可验证（存在无法计数的输出 token）")
    print(f"    QPS                     : {m['qps']:.2f} req/s")

    # ── Goodput ──
    if m.get('goodput_pct') is not None:
        print(f"\n  ── Goodput (SLO 达标率) ──")
        print(f"    SLO: TTFT ≤ {m['slo_ttft']:.3g}s AND TPOT ≤ {m['slo_tpot'] * 1000:.3g}ms")
        print(f"    达标率                  : {m['goodput_pct']:.1f}%  ({m['slo_passed']}/{m['total_requests']} 请求)")
        print(f"    Goodput QPS             : {m['goodput_qps']:.2f} req/s")

    # ── GPU / KV Cache Utilization ──
    _print_server_metrics(m.get("server_metrics", m.get("server_metrics_peak", {})))

    print("=======================================")
    return {
        "scenario": "mixed-workload",
        "metrics": metrics,
        "workload": {
            "summary": workload,
            "types": [
                {
                    "configured_input_tokens": entry["input"],
                    "configured_output_tokens": entry["output"],
                    "weight": entry["weight"],
                    "runtime": type_workloads[index],
                }
                for index, entry in enumerate(workload_entries)
            ],
        },
    }


# ── Sweep (auto-search peak throughput) ────────────────────────────────

class ApiBenchmarkSession:
    """Reusable API benchmark session: tokenizer, endpoint, and auth headers."""

    def __init__(self, args, tokenizer, url, headers):
        self.args = args
        self.tokenizer = tokenizer
        self.url = url
        self.headers = headers

    @classmethod
    def create(cls, args):
        from transformers import AutoTokenizer

        tokenizer_path = args.tokenizer or args.model
        print(f"[1/3] 加载 tokenizer: {tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        headers = {"Content-Type": "application/json"}
        if args.api_key:
            headers["Authorization"] = f"Bearer {args.api_key}"
        return cls(args, tokenizer, f"{args.api_base}/chat/completions", headers)

    def execute_batch(self, batch, concurrency):
        return run_api_benchmark_round(
            batch.prompts,
            batch.prompt_lens,
            self.url,
            self.headers,
            self.args.model,
            batch.output_lens,
            concurrency,
            self.tokenizer,
            self.args.ignore_eos,
            slo_ttft=self.args.slo_ttft,
            slo_tpot=self.args.slo_tpot,
        )


class ApiConcurrencyProbeRunner:
    """Build fresh workloads from one dataset instance for an entire sweep."""

    def __init__(self, session):
        self.session = session
        self.args = session.args
        self.requests_per_round = getattr(self.args, "sweep_requests_per_round", None)
        dataset_seed = (
            self.args.random_seed
            if self.args.random_seed is not None else self.args.seed
        )
        # A single instance keeps RandomDataset's cached prefix stable. Its RNG
        # advances in sample(), producing fresh suffixes for every batch.
        self.dataset = create_dataset(self.args.dataset, random_seed=dataset_seed)

    def request_count(self, concurrency):
        return self.requests_per_round if self.requests_per_round else max(concurrency * 2, 4)

    def build_batch(self, request_count):
        return build_request_batch(
            self.args,
            self.session.tokenizer,
            request_count,
            run_id=uuid.uuid4().hex[:8],
            dataset=self.dataset,
        )

    def prepare(self, concurrency, *, request_count=None):
        request_count = (
            self.request_count(concurrency)
            if request_count is None else request_count
        )
        if request_count < 1:
            raise ValueError(f"request_count must be positive, got {request_count}")
        batch = self.build_batch(request_count)
        return {
            "concurrency": concurrency,
            "request_count": request_count,
            "batch": batch,
            "workload": summarize_dataset_batch(batch, self.args.dataset),
        }

    def execute(self, prepared_probe):
        batch = prepared_probe["batch"]
        prepared_probe["metrics"] = self.session.execute_batch(
            batch, prepared_probe["concurrency"]
        )
        return prepared_probe

    def warmup(self):
        """Run configured, workload-matched warmup rounds outside reported results."""
        if self.args.no_warmup:
            print("[2/3] 跳过正式 workload 预热")
            return

        rounds = self.args.warmup_rounds
        # Warmup prepares the model/KV cache, not the capacity boundary. Its
        # default must therefore not inherit the formal probe rule
        # max(2 * concurrency, 4): that turns a 128K/2K warmup into four long
        # requests even at concurrency=1. Users can explicitly raise it when
        # desired, while the default stays one real workload request per round.
        configured_request_count = getattr(self.args, "warmup_requests_per_round", None)
        request_count = (
            configured_request_count
            if configured_request_count is not None else 1
        )
        concurrency = min(self.args.concurrency, request_count)
        print(
            f"[2/3] 预热正式 workload ({rounds} 轮, 每轮请求={request_count}, "
            f"执行并发={concurrency}, 总请求={rounds * request_count}) ..."
        )
        for round_index in range(1, rounds + 1):
            try:
                execution = self.execute(
                    self.prepare(concurrency, request_count=request_count)
                )
                metrics = execution["metrics"]
                successful = metrics.get("successful", "?")
                total_requests = metrics.get("total_requests", "?")
                print(
                    f"      第 {round_index}/{rounds} 轮预热完成: "
                    f"成功 {successful}/{total_requests}"
                )
            except Exception as exc:
                # Preserve the historical non-fatal warmup behavior. Formal
                # rounds still expose the error through their normal metrics.
                print(f"      第 {round_index}/{rounds} 轮预热警告: {exc}")


class SloCapacityProbeBook:
    """Record, cache, and render strict-SLO capacity probes."""

    def __init__(self, runner, args):
        self.runner = runner
        self.args = args
        self.history = []
        self.initial_probes = {}

    def probe(self, concurrency, phase, *, force=False, confirmation_attempt=None):
        if not force and concurrency in self.initial_probes:
            return self.initial_probes[concurrency]

        request_count = self.runner.request_count(concurrency)
        print(
            f"  ▶ {phase:<14s} concurrency={concurrency:>3d}  requests={request_count:>4d} ... ",
            end="", flush=True,
        )
        execution = self.runner.execute(self.runner.prepare(concurrency))
        metrics = execution["metrics"]
        workload = execution["workload"]
        required_goodput_pct = _slo_capacity_required_goodput_pct(self.args)
        passes = _slo_capacity_round_passes(
            metrics, self.args.max_failure_rate, required_goodput_pct
        )
        status = "SLO PASS" if passes else "SLO FAIL"
        total_requests = metrics.get("total_requests", request_count)
        successful = metrics.get("successful", 0)
        failed = metrics.get("failed", max(0, total_requests - successful))
        slo_passed = metrics.get("slo_passed", 0)
        goodput_pct = metrics.get("goodput_pct")
        if not _is_finite_number(goodput_pct):
            goodput_pct = (
                slo_passed / total_requests * 100 if total_requests else 0.0
            )
        avg_ttft = metrics.get("avg_ttft")
        ttft_percentiles = {
            "P50": metrics.get("p50_ttft"),
            "P90": metrics.get("p90_ttft"),
            "P95": metrics.get("p95_ttft"),
            "P99": metrics.get("p99_ttft"),
        }
        avg_tpot = metrics.get("avg_tpot")
        tpot_percentiles = {
            "P50": metrics.get("p50_tpot"),
            "P90": metrics.get("p90_tpot"),
            "P95": metrics.get("p95_tpot"),
            "P99": metrics.get("p99_tpot"),
        }
        print(f"      [{status}] phase={phase}  concurrency={concurrency}  "
              f"requests={total_requests}")
        print(
            f"        负载: Prompt {_format_token_stats(workload['prompt_tokens'])} tokens, "
            f"输出上限 {_format_token_stats(workload['requested_output_tokens'])} tokens, "
            f"共享前缀 {workload['shared_prefix_tokens']} tokens "
            f"({workload['shared_prefix_ratio'] * 100:.1f}%)"
        )
        print(
            f"        结果: 成功 {successful}/{total_requests}, 失败 {failed}, "
            f"SLO 达标 {slo_passed}/{total_requests}, "
            f"SLO 通过率/Goodput {goodput_pct:.1f}%（要求 ≥ {required_goodput_pct:.1f}%）, "
            f"失败率 {metrics.get('failure_rate', 0.0):.1%}"
        )
        print(
            "        延迟:\n"
            f"          TTFT: avg={_format_duration(avg_ttft)}, "
            f"P50={_format_duration(ttft_percentiles['P50'])}, "
            f"P90={_format_duration(ttft_percentiles['P90'])}, "
            f"P95={_format_duration(ttft_percentiles['P95'])}, "
            f"P99={_format_duration(ttft_percentiles['P99'])}\n"
            f"          TPOT: avg={_format_duration(avg_tpot, milliseconds=True)}, "
            f"P50={_format_duration(tpot_percentiles['P50'], milliseconds=True)}, "
            f"P90={_format_duration(tpot_percentiles['P90'], milliseconds=True)}, "
            f"P95={_format_duration(tpot_percentiles['P95'], milliseconds=True)}, "
            f"P99={_format_duration(tpot_percentiles['P99'], milliseconds=True)}"
        )
        print(
            f"        阈值: TTFT ≤ {self.args.slo_ttft:.3g}s, "
            f"TPOT ≤ {self.args.slo_tpot * 1000:.3g}ms, "
            f"SLO 通过率 ≥ {required_goodput_pct:.1f}%, "
            f"failure_rate ≤ {self.args.max_failure_rate:.1%}  "
            f"=> {'通过' if passes else '不通过'}"
        )
        record = {
            "phase": phase,
            "concurrency": concurrency,
            "slo_satisfied": passes,
            "failure_rate": metrics["failure_rate"],
            "goodput_pct": goodput_pct,
            "required_goodput_pct": required_goodput_pct,
            "successful": metrics["successful"],
            "total_requests": metrics["total_requests"],
            "avg_ttft": metrics.get("avg_ttft"),
            "avg_tpot": metrics.get("avg_tpot"),
            "workload": execution["workload"],
            "metrics": metrics,
        }
        if confirmation_attempt is not None:
            record["confirmation_attempt"] = confirmation_attempt
        self.history.append(record)
        if not force:
            self.initial_probes[concurrency] = record
        return record


class SloCapacityRefinementStrategy:
    """Strategy interface for post-coarse-scan SLO capacity refinement."""

    name = None

    def refine(self, probes, coarse_passing, failure_boundary):
        raise NotImplementedError

    def confirm(self, probes, last_passing):
        raise NotImplementedError


class LinearSloCapacityRefinement(SloCapacityRefinementStrategy):
    """Linearly sample a bracket, probe a bounded tail, then confirm a candidate."""

    name = "linear"

    def __init__(self, confirm_window, linear_step=1, confirm_rounds=1):
        self.confirm_window = confirm_window
        self.linear_step = linear_step
        self.confirm_rounds = confirm_rounds

    def refine(self, probes, coarse_passing, failure_boundary):
        start = coarse_passing["concurrency"] + 1
        final_scan_end = failure_boundary - 1
        print(
            f"\n  首个 SLO 失败并发为 {failure_boundary}；"
            f"在线性区间 {start}..{final_scan_end} 精扫（步长 {self.linear_step}）。"
        )
        last_passing = coarse_passing
        first_new_failure = None
        failure_confirmation_end = None
        for concurrency in range(start, failure_boundary, self.linear_step):
            record = probes.probe(concurrency, "linear")
            if record["slo_satisfied"]:
                if concurrency > last_passing["concurrency"]:
                    last_passing = record
                continue

            first_new_failure = concurrency
            # The failed probe itself has already been measured. The bounded
            # confirmation tail is intentionally sequential even when the
            # main linear scan uses a larger step, preserving the existing
            # confirm_window meaning and the step=1 probe sequence.
            failure_confirmation_end = min(
                final_scan_end, concurrency + self.confirm_window
            )
            if failure_confirmation_end > concurrency:
                print(
                    f"  并发 {concurrency} 首次失败；继续逐并发扫描 "
                    f"{concurrency + 1}..{failure_confirmation_end} "
                    f"确认窗口（{self.confirm_window} 个后续档位）。"
                )
                for tail_concurrency in range(
                    concurrency + 1, failure_confirmation_end + 1
                ):
                    tail_record = probes.probe(tail_concurrency, "linear")
                    if (
                        tail_record["slo_satisfied"]
                        and tail_concurrency > last_passing["concurrency"]
                    ):
                        last_passing = tail_record
            else:
                print(
                    f"  并发 {concurrency} 首次失败；确认窗口为 0 或已到粗扫边界，"
                    "停止线性精扫。"
                )
            break
        return {
            "last_passing": last_passing,
            "refined_failure_boundary": (
                first_new_failure
                if first_new_failure is not None else failure_boundary
            ),
            "confirmed_concurrency": None,
        }

    def confirm(self, probes, last_passing):
        return _confirm_slo_capacity_candidates(probes, self.confirm_rounds, last_passing)


def _confirm_slo_capacity_candidates(probes, confirm_rounds, last_passing):
    """Repeat candidate probes and fall back until one meets every required round."""
    if last_passing is None:
        return None, None
    candidates = sorted(
        (record for record in probes.initial_probes.values() if record["slo_satisfied"]),
        key=lambda record: record["concurrency"],
        reverse=True,
    )
    for candidate in candidates:
        confirmed = True
        for attempt in range(2, confirm_rounds + 1):
            confirmation = probes.probe(
                candidate["concurrency"],
                "confirm",
                force=True,
                confirmation_attempt=attempt,
            )
            if not confirmation["slo_satisfied"]:
                confirmed = False
                break
        if confirmed:
            return candidate, candidate["concurrency"]
    print("  候选并发未通过重复确认，未找到稳定满足 SLO 的容量。")
    return None, None


class BinaryConfirmSloCapacityRefinement(SloCapacityRefinementStrategy):
    """Binary-search a capacity candidate, inspect its neighborhood, then repeat it."""

    name = "binary-confirm"

    def __init__(self, confirm_window, confirm_rounds):
        self.confirm_window = confirm_window
        self.confirm_rounds = confirm_rounds

    def refine(self, probes, coarse_passing, failure_boundary):
        lower = coarse_passing["concurrency"]
        upper = failure_boundary
        last_passing = coarse_passing
        print(
            f"\n  首个 SLO 失败并发为 {failure_boundary}；"
            f"对区间 {lower + 1}..{upper - 1} 二分定位容量候选。"
        )
        while lower + 1 < upper:
            midpoint = (lower + upper) // 2
            record = probes.probe(midpoint, "binary")
            if record["slo_satisfied"]:
                lower = midpoint
                last_passing = record
            else:
                upper = midpoint
        window_start = max(coarse_passing["concurrency"], lower - self.confirm_window)
        window_end = min(failure_boundary - 1, lower + self.confirm_window)
        if window_start <= window_end:
            print(f"  在线性确认窗口 {window_start}..{window_end} 复核二分候选 {lower}。")
            for concurrency in range(window_start, window_end + 1):
                record = probes.probe(concurrency, "confirm-window")
                if record["slo_satisfied"] and concurrency > last_passing["concurrency"]:
                    last_passing = record
        return {
            "last_passing": last_passing,
            "refined_failure_boundary": upper,
            "confirmed_concurrency": None,
        }

    def confirm(self, probes, last_passing):
        return _confirm_slo_capacity_candidates(probes, self.confirm_rounds, last_passing)


def create_slo_capacity_refinement_strategy(args):
    """Factory for the selected SLO capacity refinement policy."""
    strategy_name = getattr(args, "slo_capacity_search_strategy", "binary-confirm")
    if strategy_name == "linear":
        return LinearSloCapacityRefinement(
            getattr(args, "slo_capacity_confirm_window", 8),
            getattr(args, "slo_capacity_linear_step", 1),
            getattr(args, "slo_capacity_confirm_rounds", 3),
        )
    if strategy_name == "binary-confirm":
        return BinaryConfirmSloCapacityRefinement(
            getattr(args, "slo_capacity_confirm_window", 8),
            getattr(args, "slo_capacity_confirm_rounds", 3),
        )
    raise ValueError(f"unsupported SLO capacity refinement strategy: {strategy_name}")


def run_sweep_benchmark(args):
    """Search peak throughput across powers-of-two concurrency levels.

    Levels are measured as 1, 2, 4, ... up to ``sweep_max_concurrency``. The
    search stops after an eligible, verifiable level improves on the prior best
    by less than 5%; this is the established fast plateau heuristic, not an
    exhaustive search. The reported peak is the highest eligible phase rate.
    """
    sweep_type = None
    metric_key = None
    metric_label = None
    max_concurrency = getattr(args, "sweep_max_concurrency", 128)
    plateau_threshold = 0.05

    print("[Sweep] 搜索最大吞吐量（2 的幂并发扫描，阶段由实际请求输出上限判定）")
    print(f"        模型: {args.model}")
    print(f"        数据集: {args.dataset}")
    print(f"        搜索范围: 1, 2, 4, ... ≤ {max_concurrency}")
    print(f"        平台停止条件: 合格档位相对最佳吞吐提升 < {plateau_threshold:.0%}")
    print(f"        候选门槛: failure_rate ≤ {args.max_failure_rate:.1%}")
    print()

    session = ApiBenchmarkSession.create(args)
    probe_runner = ApiConcurrencyProbeRunner(session)
    probe_runner.warmup()

    print("[3/3] 开始扫描（首轮将根据实际输出上限选择 Prefill 或 Decode 指标）")
    print()

    sweep_history = []
    best_throughput = None
    best_concurrency = None
    concurrency = 1
    while concurrency <= max_concurrency:
        probe_execution = probe_runner.prepare(concurrency)
        n_requests = probe_execution["request_count"]
        batch = probe_execution["batch"]
        workload = probe_execution["workload"]
        all_prefill = all(limit == 1 for limit in batch.output_lens)
        all_decode = all(limit > 1 for limit in batch.output_lens)
        if not all_prefill and not all_decode:
            raise ValueError(
                "sweep batch mixes one-token prefill-only requests with multi-token "
                "decode requests; use --random-range-ratio values that keep all "
                "requested outputs at 1 or all above 1"
            )
        round_sweep_type = "prefill" if all_prefill else "decode"
        if sweep_type is None:
            sweep_type = round_sweep_type
            metric_key = "prefill_throughput" if sweep_type == "prefill" else "decode_throughput"
            metric_label = (
                "Prefill 阶段吞吐量 (tokens/s)"
                if sweep_type == "prefill"
                else "Decode 阶段吞吐量 (tokens/s)"
            )
            print(
                f"      实际负载: Prompt={_format_token_stats(workload['prompt_tokens'])} tokens; "
                f"请求输出上限={_format_token_stats(workload['requested_output_tokens'])} tokens"
            )
            print(f"      目标指标: {metric_label}")
        elif round_sweep_type != sweep_type:
            raise ValueError(
                "sweep batch output limits crossed the Prefill/Decode boundary; "
                "use --random-range-ratio values that keep all requested outputs "
                "at 1 or all above 1"
            )

        print(f"  ▶ concurrency={concurrency:>3d}  requests={n_requests:>4d} ... ", end="", flush=True)
        metrics = probe_runner.execute(probe_execution)["metrics"]
        throughput = metrics.get(metric_key)
        failure_rate = metrics["failure_rate"]
        avg_ttft = metrics.get("avg_ttft")
        eligible = (
            bool(metrics["successful"])
            and _is_finite_number(throughput)
            and failure_rate <= args.max_failure_rate
        )
        previous_best = best_throughput
        improvement = None
        if eligible and previous_best is not None:
            improvement = (
                (throughput - previous_best) / max(previous_best, 1.0)
            )

        if not metrics["successful"]:
            status = "FAILED (all requests failed)"
        elif not _is_finite_number(throughput):
            status = "UNVERIFIABLE (target token throughput unavailable)"
        elif failure_rate > args.max_failure_rate:
            status = (
                f"DISQUALIFIED (failure {failure_rate:.1%} > "
                f"limit {args.max_failure_rate:.1%})"
            )
        elif previous_best is None or throughput > previous_best:
            best_throughput = throughput
            best_concurrency = concurrency
            status = "★ NEW BEST"
        else:
            status = "eligible"

        stopped_on_plateau = (
            improvement is not None
            and improvement < plateau_threshold
            and throughput <= best_throughput
        )
        if stopped_on_plateau:
            status += f" → plateau (< {plateau_threshold:.0%}; stopping)"

        sweep_history.append({
            "concurrency": concurrency,
            "throughput": throughput if _is_finite_number(throughput) else None,
            "avg_ttft": avg_ttft,
            "failed_ratio": failure_rate,
            "failure_rate": failure_rate,
            "goodput_pct": metrics["goodput_pct"],
            "qps": metrics["qps"],
            "eligible": eligible,
            "improvement_pct": improvement * 100 if improvement is not None else None,
            "stopped_on_plateau": stopped_on_plateau,
            "workload": workload,
            "status": status,
        })
        throughput_text = f"{throughput:.1f}" if _is_finite_number(throughput) else "n/a"
        avg_ttft_text = f"{avg_ttft:.3f}s" if _is_finite_number(avg_ttft) else "n/a"
        print(
            f"{metric_label}: {throughput_text:>10s}  avg_ttft: {avg_ttft_text:>8s}  "
            f"failed: {failure_rate:.0%}  {status}"
        )
        if stopped_on_plateau:
            break
        concurrency *= 2

    print()
    print("=" * 96)
    print(
        f"{'Concurrency':>12s}  {'Throughput (tokens/s)':>22s}  {'Avg TTFT (s)':>13s}  "
        f"{'QPS':>8s}  {'Failed':>7s}  {'Eligible':>8s}  Status"
    )
    print("-" * 96)
    for history in sweep_history:
        marker = "  ←" if history["concurrency"] == best_concurrency else ""
        throughput_text = (
            f"{history['throughput']:.1f}"
            if _is_finite_number(history["throughput"])
            else "n/a"
        )
        avg_ttft_text = (
            f"{history['avg_ttft']:.3f}"
            if _is_finite_number(history["avg_ttft"])
            else "n/a"
        )
        eligible_text = "yes" if history["eligible"] else "no"
        print(
            f"{history['concurrency']:>12d}  {throughput_text:>22s}  {avg_ttft_text:>13s}  "
            f"{history['qps']:>8.2f}  {history['failed_ratio']*100:>6.0f}%  "
            f"{eligible_text:>8s}  {history['status']}{marker}"
        )
    print("=" * 96)
    if best_concurrency is not None:
        stop_reason = "（因平台条件提前停止）" if sweep_history[-1]["stopped_on_plateau"] else ""
        print(
            f"\n✅ 最佳合格配置: concurrency = {best_concurrency}, "
            f"峰值 {metric_label}: {best_throughput:.1f} {stop_reason}"
        )
    else:
        print("\n❌ 未找到合格的可验证配置：所有档位均失败、不可验证或超过失败率门槛")

    return {
        "scenario": "sweep",
        "sweep_type": sweep_type,
        "metric": metric_key,
        "search_strategy": "powers_of_two_until_plateau",
        "max_concurrency": max_concurrency,
        "best_throughput": best_throughput,
        "best_concurrency": best_concurrency,
        "best_workload": next(
            (history["workload"] for history in sweep_history
             if history["concurrency"] == best_concurrency),
            None,
        ),
        "history": sweep_history,
    }


def _slo_capacity_required_goodput_pct(args):
    """Return the capacity-search SLO pass-rate threshold as a percentage.

    ``min_goodput_pct`` historically used zero to disable the suite-level
    quality gate. Preserve the capacity search's previous strict behavior in
    that case; a positive value explicitly relaxes its per-probe boundary.
    """
    return args.min_goodput_pct if args.min_goodput_pct > 0 else 100.0


def _slo_capacity_round_passes(metrics, max_failure_rate, min_goodput_pct):
    """Return whether one probe meets Goodput and failure-rate thresholds."""
    total_requests = metrics.get("total_requests")
    goodput_pct = metrics.get("goodput_pct")
    return (
        isinstance(total_requests, int)
        and total_requests > 0
        and _is_finite_number(goodput_pct)
        and 0.0 <= goodput_pct <= 100.0
        and goodput_pct >= min_goodput_pct
        and _is_finite_number(metrics.get("failure_rate"))
        and metrics["failure_rate"] <= max_failure_rate
    )


def run_slo_capacity_benchmark(args):
    """Find the highest concurrency where every request satisfies the SLO.

    Both strategies start with powers-of-two probes. ``linear`` samples the
    final bracket at a configurable positive step (one by default) for an
    exhaustive, noise-tolerant result at the default. ``binary-confirm`` uses
    binary search, checks a small window around the candidate, then repeats
    the selected candidate before reporting it.
    """
    max_concurrency = getattr(args, "sweep_max_concurrency", 128)
    required_goodput_pct = _slo_capacity_required_goodput_pct(args)
    refinement_strategy = create_slo_capacity_refinement_strategy(args)
    print("[SLO Capacity] 搜索满足 SLO 的最大并发")
    print(f"               模型: {args.model}")
    print(f"               数据集: {args.dataset}")
    print(f"               粗扫: 1, 2, 4, ... ≤ {max_concurrency}")
    if refinement_strategy.name == "linear":
        print(
            "               精扫: linear（步长 "
            f"{refinement_strategy.linear_step}；首次失败后额外逐并发确认 "
            f"{refinement_strategy.confirm_window} 个后续档位；最终候选共 "
            f"{refinement_strategy.confirm_rounds} 轮）"
        )
    else:
        print(
            f"               精扫: binary-confirm（二分 + ±{refinement_strategy.confirm_window} 并发确认窗口，"
            f"候选共 {refinement_strategy.confirm_rounds} 轮）"
        )
    print(
        f"               通过条件: 单请求 TTFT ≤ {args.slo_ttft:.3g}s，"
        f"TPOT ≤ {args.slo_tpot * 1000:.3g}ms；"
        f"SLO 通过率 ≥ {required_goodput_pct:.1f}%，"
        f"failure_rate ≤ {args.max_failure_rate:.1%}"
    )
    print()

    session = ApiBenchmarkSession.create(args)
    probe_runner = ApiConcurrencyProbeRunner(session)
    print("      工作负载: 每轮从同一数据集实例生成新的请求 batch")
    probe_runner.warmup()
    probes = SloCapacityProbeBook(probe_runner, args)

    print("[3/3] 开始 SLO 容量搜索 ...")
    last_passing = None
    failure_boundary = None
    concurrency = 1
    while concurrency <= max_concurrency:
        record = probes.probe(concurrency, "coarse")
        if record["slo_satisfied"]:
            last_passing = record
            concurrency *= 2
        else:
            failure_boundary = concurrency
            break

    coarse_passing = last_passing
    refined_failure_boundary = failure_boundary
    confirmed_concurrency = None
    if failure_boundary is not None and coarse_passing is not None:
        refinement = refinement_strategy.refine(probes, coarse_passing, failure_boundary)
        last_passing = refinement["last_passing"]
        refined_failure_boundary = refinement["refined_failure_boundary"]
    elif failure_boundary == 1:
        print("\n  并发 1 已不满足 SLO，未找到可用容量。")
    elif last_passing is not None:
        print(
            f"\n  未在 2 的幂粗扫范围内观察到 SLO 失败；"
            f"最大已验证并发为 {last_passing['concurrency']}。"
        )

    last_passing, confirmed_concurrency = refinement_strategy.confirm(probes, last_passing)

    print("\n" + "=" * 84)
    print(f"{'Phase':>14s}  {'Concurrency':>12s}  {'SLO Pass':>10s}  {'Failed':>8s}  {'Status'}")
    print("-" * 84)
    for record in probes.history:
        marker = "  ←" if record is last_passing else ""
        status = "SLO PASS" if record["slo_satisfied"] else "SLO FAIL"
        print(
            f"{record['phase']:>14s}  {record['concurrency']:>12d}  "
            f"{record['goodput_pct']:>9.1f}%  "
            f"{record['failure_rate']*100:>7.1f}%  "
            f"{status}{marker}"
        )
    print("=" * 84)
    if last_passing is not None:
        print(
            f"\n✅ 满足 SLO 通过率阈值的最大并发: {last_passing['concurrency']} "
            f"(Goodput {last_passing['goodput_pct']:.1f}%, "
            f"要求 ≥ {required_goodput_pct:.1f}%)"
        )
    else:
        print("\n❌ 未找到满足全部请求 SLO 的并发配置")

    return {
        "scenario": "slo-capacity-search",
        "search_strategy": (
            "powers_of_two_then_linear_refinement"
            if refinement_strategy.name == "linear"
            else "powers_of_two_then_binary_confirmation"
        ),
        "refinement_strategy": refinement_strategy.name,
        "linear_step": (
            refinement_strategy.linear_step
            if refinement_strategy.name == "linear" else None
        ),
        "confirmation_window": getattr(refinement_strategy, "confirm_window", None),
        "confirmation_rounds": getattr(refinement_strategy, "confirm_rounds", None),
        "max_concurrency": max_concurrency,
        "required_goodput_pct": required_goodput_pct,
        "max_failure_rate": args.max_failure_rate,
        "failure_boundary": failure_boundary,
        "refined_failure_boundary": refined_failure_boundary,
        "confirmed_concurrency": confirmed_concurrency,
        "max_passing_concurrency": last_passing["concurrency"] if last_passing else None,
        # Match the normal benchmark result contract: consumers can always
        # read the selected round's full aggregate data from result.metrics.
        # Keep selected_metrics for backward compatibility with existing SLO
        # reports and quality-gate handling.
        "metrics": last_passing["metrics"] if last_passing else None,
        "selected_metrics": last_passing["metrics"] if last_passing else None,
        "selected_workload": last_passing["workload"] if last_passing else None,
        "workload_policy": {
            "policy": "fresh_batches_from_shared_dataset_instance",
            "random_seed": (
                args.random_seed if args.random_seed is not None else args.seed
            ),
            "shared_prefix_cached": (
                args.dataset == "random" and args.random_prefix_len > 0
            ),
        },
        "history": probes.history,
    }


# ── PD Ratio Calculator ───────────────────────────────────────────────

def run_pd_ratio_benchmark(args):
    """
    Automatically benchmark prefill and decode throughput on a single instance,
    then calculate the recommended P:D disaggregation ratio based on the expected workload.
    """
    from transformers import AutoTokenizer
    import math

    avg_input = getattr(args, "avg_input_tokens", 4096)
    avg_output = getattr(args, "avg_output_tokens", 2048)
    total_gpus = getattr(args, "total_gpus", None)

    tokenizer_path = args.tokenizer or args.model
    print("=" * 70)
    print("  PD 分离比例自动计算")
    print("=" * 70)
    print(f"  模型: {args.model}")
    print(f"  业务负载: 平均输入 {avg_input} tokens, 平均输出 {avg_output} tokens")
    if total_gpus:
        print(f"  总 GPU 数: {total_gpus}")
    print()

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    url = f"{args.api_base}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    # Warmup
    if not args.no_warmup:
        print("[1/4] 预热服务 ...")
        warmup_payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": "你好"}],
            "max_tokens": 10, "temperature": 0, "stream": False,
        }
        try:
            requests.post(url, json=warmup_payload, headers=headers, timeout=120).raise_for_status()
            print("      预热完成")
        except Exception as e:
            print(f"      预热警告: {e}")

    # ── Phase 1: Measure Prefill throughput ──
    run_uuid = uuid.uuid4().hex[:8]
    n_prefill = 8
    prefill_batch = build_request_batch(
        args, tokenizer, n_prefill, input_len=avg_input, output_len=1,
        share_prefix=False, random_range_ratio=0.0, run_id=run_uuid,
    )
    prefill_prompts = prefill_batch.prompts
    prefill_workload = summarize_dataset_batch(prefill_batch, args.dataset)
    print(
        f"\n[2/4] 测量 Prefill 吞吐量 (实际 Prompt="
        f"{_format_token_stats(prefill_workload['prompt_tokens'])}, 请求输出上限="
        f"{_format_token_stats(prefill_workload['requested_output_tokens'])}, concurrency=4) ..."
    )

    prefill_metrics = run_api_benchmark_round(
        prefill_prompts, prefill_batch.prompt_lens, url, headers, args.model,
        prefill_batch.output_lens, 4, tokenizer, True,
        slo_ttft=args.slo_ttft, slo_tpot=args.slo_tpot
    )
    if not prefill_metrics["successful"]:
        print("ERROR: Prefill 测试失败")
        return None
    prefill_throughput = prefill_metrics["prefill_throughput"]
    if not _is_finite_number(prefill_throughput) or prefill_throughput <= 0:
        print("ERROR: Prefill 吞吐量不可验证，无法计算 PD 比例")
        return None
    prefill_avg_ttft = prefill_metrics["avg_ttft"]
    prefill_p99_ttft = prefill_metrics["p99_ttft"]
    print(f"      Prefill 吞吐量: {prefill_throughput:.1f} tokens/s")
    print(f"      平均 TTFT     : {prefill_avg_ttft * 1000:.1f} ms")
    print(f"      P99  TTFT     : {prefill_p99_ttft * 1000:.1f} ms")

    # ── Phase 2: Measure Decode throughput ──
    run_uuid = uuid.uuid4().hex[:8]
    n_decode = 8
    decode_batch = build_request_batch(
        args, tokenizer, n_decode, input_len=128, output_len=min(avg_output, 4096),
        share_prefix=False, random_range_ratio=0.0, run_id=run_uuid,
    )
    decode_prompts = decode_batch.prompts
    decode_workload = summarize_dataset_batch(decode_batch, args.dataset)
    print(
        f"\n[3/4] 测量 Decode 吞吐量 (实际 Prompt="
        f"{_format_token_stats(decode_workload['prompt_tokens'])}, 请求输出上限="
        f"{_format_token_stats(decode_workload['requested_output_tokens'])}, concurrency=4) ..."
    )

    decode_metrics = run_api_benchmark_round(
        decode_prompts, decode_batch.prompt_lens, url, headers, args.model,
        decode_batch.output_lens, 4, tokenizer, True,
        slo_ttft=args.slo_ttft, slo_tpot=args.slo_tpot
    )
    if not decode_metrics["successful"]:
        print("ERROR: Decode 测试失败")
        return None
    decode_throughput = decode_metrics["decode_throughput"]
    if not _is_finite_number(decode_throughput) or decode_throughput <= 0:
        print("ERROR: Decode 吞吐量不可验证（需要所有成功请求具有可验证的输出 token 数），无法计算 PD 比例")
        return None
    decode_avg_tpot = decode_metrics.get("avg_tpot")
    decode_avg_estimated_itl = decode_metrics.get("avg_estimated_itl")
    print(f"      Decode 吞吐量: {decode_throughput:.1f} tokens/s")
    if decode_avg_tpot:
        print(f"      平均 TPOT     : {decode_avg_tpot * 1000:.1f} ms")
    if decode_avg_estimated_itl:
        print(f"      预估 ITL      : {decode_avg_estimated_itl * 1000:.1f} ms")

    # ── Phase 3: Calculate optimal P:D ratio ──
    print(f"\n[4/4] 计算最佳 PD 比例 ...")

    prefill_time = avg_input / max(prefill_throughput, 1.0)
    decode_time = avg_output / max(decode_throughput, 1.0)
    total_time = prefill_time + decode_time

    prefill_pct = prefill_time / total_time * 100
    decode_pct = decode_time / total_time * 100

    # Raw ratio
    raw_ratio = prefill_time / decode_time if decode_time > 0 else float('inf')

    # Find closest simple integer ratio (e.g., 1:3, 2:5, 1:10)
    def simplify_ratio(p_time, d_time, max_total=20):
        """Find the simplest integer P:D ratio."""
        if p_time <= 0:
            return 0, 1
        ratio = d_time / p_time  # decode instances per 1 prefill
        best_p, best_d = 1, max(1, round(ratio))
        best_err = abs(best_d / best_p - ratio)
        for p in range(1, max_total):
            d = max(1, round(ratio * p))
            if p + d > max_total:
                break
            err = abs(d / p - ratio)
            if err < best_err:
                best_err = err
                best_p, best_d = p, d
        return best_p, best_d

    p_count, d_count = simplify_ratio(prefill_time, decode_time)

    # ── Calculate optimal max-num-seqs / max-num-batched-tokens ──
    # Prefill instance:
    #   - Few concurrent requests (large context each), focus on chunked prefill
    #   - max-num-seqs: enough to keep GPU busy but not too many (memory pressure)
    #   - max-num-batched-tokens: sized for chunked prefill of large contexts
    prefill_seqs_per_sec = prefill_throughput / max(avg_input, 1)
    # Allow ~2-4 seconds worth of requests to be queued
    prefill_max_num_seqs = max(4, min(32, int(prefill_seqs_per_sec * 2)))
    # Round to power of 2 or nice number
    for nice in [4, 8, 16, 32]:
        if prefill_max_num_seqs <= nice:
            prefill_max_num_seqs = nice
            break
    # batched tokens: allow processing a full context per step
    prefill_max_batched_tokens = max(avg_input, 8192)
    # Round up to nearest power of 2
    for nice in [8192, 16384, 32768, 65536, 131072, 262144]:
        if prefill_max_batched_tokens <= nice:
            prefill_max_batched_tokens = nice
            break

    # Decode instance:
    #   - Many concurrent requests (each generates 1 token per step)
    #   - max-num-seqs: high to maximize batching → higher throughput
    #   - max-num-batched-tokens: must accommodate all seqs' decode tokens
    decode_avg_tpot = decode_metrics.get("avg_tpot")
    if decode_avg_tpot and decode_avg_tpot > 0:
        # Optimal batch size ≈ throughput * time_per_step
        # Each step: batch_size requests processed in ~TPOT time
        optimal_decode_batch = int(decode_throughput * decode_avg_tpot)
        decode_max_num_seqs = max(32, min(512, optimal_decode_batch))
    else:
        # Fallback: estimate from throughput (assume ~10ms TPOT)
        decode_max_num_seqs = max(32, min(512, int(decode_throughput * 0.01)))
    # Round to nice number
    for nice in [32, 48, 64, 96, 128, 192, 256, 384, 512]:
        if decode_max_num_seqs <= nice:
            decode_max_num_seqs = nice
            break
    # Decode batched tokens: at least max_num_seqs (1 token/seq per step) + headroom for scheduling
    decode_max_batched_tokens = max(decode_max_num_seqs * 2, 8192)
    for nice in [8192, 16384, 32768, 65536]:
        if decode_max_batched_tokens <= nice:
            decode_max_batched_tokens = nice
            break

    print()
    print("=" * 70)
    print("  PD 分离比例分析结果")
    print("=" * 70)
    print()
    print("  ── 吞吐量指标 ──")
    print(f"    Prefill 吞吐量 (单实例)  : {prefill_throughput:>12.1f} tokens/s")
    print(f"    Decode  吞吐量 (单实例)  : {decode_throughput:>12.1f} tokens/s")
    print()
    print("  ── 延迟指标 ──")
    print(f"    Prefill 平均 TTFT        : {prefill_avg_ttft*1000:>10.1f} ms")
    print(f"    Prefill P99  TTFT        : {prefill_p99_ttft*1000:>10.1f} ms")
    if decode_avg_tpot:
        print(f"    Decode  平均 TPOT        : {decode_avg_tpot*1000:>10.1f} ms")
    if decode_avg_estimated_itl:
        print(f"    Decode  预估 ITL         : {decode_avg_estimated_itl*1000:>10.1f} ms")
    print()
    print("  ── 业务负载时间分析（按配置的平均输入/输出计算） ──")
    print(f"    每请求 Prefill 耗时       : {prefill_time*1000:>10.1f} ms  ({prefill_pct:.1f}%)")
    print(f"    每请求 Decode  耗时       : {decode_time*1000:>10.1f} ms  ({decode_pct:.1f}%)")
    print(f"    每请求总耗时 (P+D)        : {total_time*1000:>10.1f} ms")

    # ── Determine if PD disaggregation is recommended ──
    # Reasons NOT to use PD:
    #   1. One phase dominates overwhelmingly (>95%) → separating wastes GPUs on the minor phase
    #   2. P:D ratio is close to 1:1 → no scheduling benefit, adds KV transfer overhead
    #   3. Total GPU count too small to support separate instances
    #   4. KV transfer overhead may negate the benefit (estimated ~10-20% overhead)
    pd_reasons_against = []
    pd_reasons_for = []
    pd_recommended = True

    if prefill_pct > 95:
        pd_reasons_against.append(f"Prefill 占比过高 ({prefill_pct:.1f}%)，Decode 几乎不耗时，分离后 Decode 实例会空转浪费 GPU")
        pd_recommended = False
    elif decode_pct > 98:
        pd_reasons_against.append(f"Decode 占比过高 ({decode_pct:.1f}%)，Prefill 极快，分离收益极小但增加了 KV 传输开销")
        pd_recommended = False

    if 0.7 <= raw_ratio <= 1.4:
        pd_reasons_against.append(f"P:D 耗时比接近 1:1 ({raw_ratio:.2f})，分离后两边负载相近，无法通过不同配比优化")
        pd_recommended = False

    # Latency-based factors
    # High P99 TTFT relative to avg indicates prefill-decode contention
    if prefill_p99_ttft > prefill_avg_ttft * 2.5 and prefill_p99_ttft > 0.5:
        pd_reasons_for.append(f"P99 TTFT ({prefill_p99_ttft*1000:.0f}ms) 远高于平均 ({prefill_avg_ttft*1000:.0f}ms)，"
                              f"存在 Prefill-Decode 资源争抢，PD 分离可显著降低尾部延迟")

    # High TTFT in general → prefill is bottleneck, separation helps
    if prefill_avg_ttft > 1.0:
        pd_reasons_for.append(f"平均 TTFT 较高 ({prefill_avg_ttft*1000:.0f}ms)，独立 Prefill 实例可消除 Decode 干扰，降低首 Token 延迟")

    if total_gpus:
        tp_size = getattr(args, "tp_size", 2)
        gpus_per_instance = tp_size
        total_instances = total_gpus // gpus_per_instance
        if total_instances < 3:
            pd_reasons_against.append(f"GPU 数量不足: {total_gpus} 卡 / TP={tp_size} = {total_instances} 实例，PD 分离至少需要 3 实例 (1P+1D+router)")
            pd_recommended = False

    print()
    if pd_recommended:
        print(f"  ★ 建议: 启用 PD 分离")
        print(f"  ★ 建议 PD 比例              : {p_count}P : {d_count}D")
        if pd_reasons_for:
            print()
            print("  ── 分离收益分析 ──")
            for reason in pd_reasons_for:
                print(f"    ✓ {reason}")
        print()
        print("  ── Prefill 实例建议参数 ──")
        print(f"    max-num-seqs            : {prefill_max_num_seqs}")
        print(f"    max-num-batched-tokens  : {prefill_max_batched_tokens}")
        print()
        print("  ── Decode 实例建议参数 ──")
        print(f"    max-num-seqs            : {decode_max_num_seqs}")
        print(f"    max-num-batched-tokens  : {decode_max_batched_tokens}")

        if total_gpus:
            scale = total_instances / (p_count + d_count)
            actual_p = max(1, round(p_count * scale))
            actual_d = total_instances - actual_p
            if actual_d < 1:
                actual_d = 1
                actual_p = total_instances - actual_d
            print()
            print(f"  ★ {total_gpus} 卡实际部署 (TP={tp_size}) : {actual_p}P : {actual_d}D  (共 {actual_p + actual_d} 实例)")
            print()
            print(f"  建议启动命令:")
            print(f"    PD_MODE=enabled PD_PREFILL_COUNT={actual_p} PD_DECODE_COUNT={actual_d} \\")
            print(f"    PD_PREFILL_MAX_NUM_SEQS={prefill_max_num_seqs} \\")
            print(f"    PD_PREFILL_MAX_NUM_BATCHED_TOKENS={prefill_max_batched_tokens} \\")
            print(f"    PD_DECODE_MAX_NUM_SEQS={decode_max_num_seqs} \\")
            print(f"    PD_DECODE_MAX_NUM_BATCHED_TOKENS={decode_max_batched_tokens} \\")
            print(f"    TP_SIZE={tp_size} bash docker-start-deepseek.sh")
    else:
        print(f"  ★ 建议: 不启用 PD 分离，使用统一部署")
        print()
        print("  ── 不建议分离的原因 ──")
        for reason in pd_reasons_against:
            print(f"    ⚠ {reason}")
        if pd_reasons_for:
            print()
            print("  ── 但以下延迟特征可能受益于分离 ──")
            for reason in pd_reasons_for:
                print(f"    ℹ {reason}")
        print()
        print(f"  统一部署可以避免 KV cache 传输开销 (~10-20%)、Router 调度延迟，")
        print(f"  并简化运维。当负载分布不均衡或 GPU 资源有限时，统一部署通常更优。")
        if total_gpus:
            print()
            print(f"  建议启动命令 (统一部署, {total_instances} 实例):")
            print(f"    TP_SIZE={tp_size} bash docker-start-deepseek.sh")

    print("=" * 70)
    return {
        "scenario": "pd-ratio",
        "prefill": prefill_metrics,
        "decode": decode_metrics,
        "prefill_workload": prefill_workload,
        "decode_workload": decode_workload,
        "analysis": {
            "business_workload": {
                "configured_avg_input_tokens": avg_input,
                "configured_avg_output_tokens": avg_output,
            },
            # Legacy aliases retained for consumers of older reports.
            "avg_input_tokens": avg_input,
            "avg_output_tokens": avg_output,
            "prefill_time_seconds": prefill_time,
            "decode_time_seconds": decode_time,
            "prefill_share_pct": prefill_pct,
            "decode_share_pct": decode_pct,
            "recommended": pd_recommended,
            "recommended_ratio": {"prefill": p_count, "decode": d_count},
            "reasons_for": pd_reasons_for,
            "reasons_against": pd_reasons_against,
            "prefill_max_num_seqs": prefill_max_num_seqs,
            "prefill_max_num_batched_tokens": prefill_max_batched_tokens,
            "decode_max_num_seqs": decode_max_num_seqs,
            "decode_max_num_batched_tokens": decode_max_batched_tokens,
        },
    }


# ── Preset Test Cases ──────────────────────────────────────────────────
# To add a new preset: append a new entry to PRESET_REGISTRY.
# Each preset specifies:
#   name        - CLI identifier (e.g. "prefill-max")
#   description - one-line description shown in help
#   focus       - key metrics to watch
#   params      - dict of arg overrides (only applied when CLI arg is None)
#   sweep       - if True, run in auto-search mode instead of single round

from dataclasses import dataclass, field
from typing import Dict, Any, List


@dataclass
class Preset:
    """A benchmark preset that pre-configures optimal parameters for a specific test scenario."""
    name: str
    description: str
    focus: str
    params: Dict[str, Any] = field(default_factory=dict)
    sweep: bool = False
    slo_capacity_search: bool = False
    pd_ratio: bool = False
    mixed_workload: bool = False


# ── Preset Registry (add new presets here) ─────────────────────────────
PRESET_REGISTRY: List[Preset] = [
    Preset(
        name="prefill-max",
        description="测试系统最大 Prefill 吞吐能力",
        focus="Prefill 系统总吞吐量, 平均/P99 TTFT",
        params={
            "context_len": 131072,
            "max_tokens": 1,
            "concurrency": 4,
            "share_prefix": False,
        },
    ),
    Preset(
        name="decode-max",
        description="测量当前请求形状场景下的 Decode 阶段速率",
        focus="Decode 阶段吞吐量, TPOT, 预估 ITL",
        params={
            # 128K/4096 is the default scenario only. CLI/config token fields
            # applied after the preset deliberately override this workload.
            "context_len": 131072,
            "max_tokens": 4096,
            "concurrency": 8,
            "share_prefix": False,
        },
    ),
    Preset(
        name="mixed",
        description="模拟真实混合负载 (Prefill + Decode)",
        focus="综合吞吐量, QPS, 端到端延迟",
        params={
            "context_len": 8192,
            "max_tokens": 1024,
            "concurrency": 4,
            "share_prefix": False,
        },
    ),
    Preset(
        name="prefill-sweep",
        description="自动搜索系统最大 Prefill 吞吐量 (逐步增加并发)",
        focus="峰值 Prefill 吞吐量, 最佳并发数",
        params={
            "dataset": "random",
            "context_len": 131072,
            "max_tokens": 1,
            "share_prefix": False,
        },
        sweep=True,
    ),
    Preset(
        name="decode-sweep",
        description="按 2 的幂并发扫描当前场景的最大 Decode 阶段速率",
        focus="合格轮次中的峰值 Decode 阶段吞吐量, 最佳并发数",
        params={
            "dataset": "random",
            # Keep the 128K/4096 default scenario configurable through the
            # normal --context-len/--max-tokens or suite tokens overrides.
            "context_len": 131072,
            "max_tokens": 4096,
            "share_prefix": False,
        },
        sweep=True,
    ),
    Preset(
        name="slo-capacity-sweep",
        description="按 SLO 搜索当前场景可承载的最大并发",
        focus="最大 SLO 合格并发数, TTFT, TPOT, Goodput",
        params={
            "dataset": "random",
            # This 128K/4096 workload is a default scenario. CLI and suite
            # token parameters are applied after the preset and may override it.
            "context_len": 131072,
            "max_tokens": 4096,
            "share_prefix": False,
        },
        slo_capacity_search=True,
    ),
    Preset(
        name="pd-ratio",
        description="自动测量 P/D 吞吐量并计算最佳 PD 分离比例",
        focus="最佳 P:D 比例, 部署建议",
        params={
            "share_prefix": False,
        },
        pd_ratio=True,
    ),
    Preset(
        name="mixed-workload",
        description="混合负载测试 (不同长短上下文组合)",
        focus="综合吞吐量, QPS, 各类请求延迟分布",
        params={
            "concurrency": 8,
            "num_prompts": 32,
            "share_prefix": False,
            "workload_mix": "128:4096:0.3,4096:1024:0.4,32768:256:0.2,131072:1:0.1",
        },
        mixed_workload=True,
    ),
]


def get_preset_names() -> List[str]:
    """Return all registered preset names for argparse choices."""
    return [p.name for p in PRESET_REGISTRY]


def get_preset_by_name(name: str) -> Preset:
    """Look up a preset by name. Raises ValueError if not found."""
    for p in PRESET_REGISTRY:
        if p.name == name:
            return p
    raise ValueError(f"Unknown preset: {name}. Available: {get_preset_names()}")


def build_preset_epilog() -> str:
    """Generate the --help epilog text from the registry."""
    lines = ["预设测试用例 (--preset):"]
    for p in PRESET_REGISTRY:
        params_desc = ", ".join(f"{k}={v}" for k, v in p.params.items())
        sweep_tag = " [自动搜索]" if p.sweep else ""
        lines.append(f"  {p.name:<16s}{p.description}{sweep_tag}")
        lines.append(f"                    → {params_desc}")
        lines.append(f"                    → 关注指标: {p.focus}")
        lines.append("")
    lines.append("示例:")
    lines.append("  python benchmark.py --mode api --preset prefill-max -c 4 --api-base http://localhost:8001/v1")
    lines.append("  python benchmark.py --mode api --preset decode-sweep --api-base http://localhost:8001/v1")
    return "\n".join(lines)


def apply_preset(args, verbose: bool = True) -> None:
    """Apply preset defaults to args without overriding explicitly supplied values."""
    if not args.preset:
        return
    preset = get_preset_by_name(args.preset)
    if verbose:
        print(f"[Preset] 使用预设测试用例: {preset.name} — {preset.description}")
        print(f"         关注指标: {preset.focus}")
    for key, value in preset.params.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
            if verbose:
                print(f"         {key} = {value}")
    args._sweep = preset.sweep
    args._slo_capacity_search = preset.slo_capacity_search
    args._pd_ratio = preset.pd_ratio
    args._mixed_workload = preset.mixed_workload
    if verbose:
        print()


def apply_final_defaults(args) -> None:
    """Fill in remaining None values with global defaults (when no preset was used or preset didn't set them)."""
    defaults = {
        "dataset": "text",
        "context_len": 131072,
        "max_tokens": 1,
        "concurrency": 1,
        "share_prefix": False,
    }
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if not hasattr(args, "_sweep"):
        args._sweep = False
    if not hasattr(args, "_slo_capacity_search"):
        args._slo_capacity_search = False
    if not hasattr(args, "_pd_ratio"):
        args._pd_ratio = False
    if not hasattr(args, "_mixed_workload"):
        args._mixed_workload = False


# ── Configurable Benchmark Suites / JSON Reports ──────────────────────

SUITE_SCHEMA_VERSION = 1
SUITE_SCENARIOS = {"single", "mixed-workload", "sweep", "slo-capacity-search", "pd-ratio"}
SUITE_MANAGEMENT_ARGS = {
    "config", "report", "case_filters", "tag_filters", "list_cases",
    "validate_config", "fail_fast", "no_resume",
}


class BenchmarkConfigError(ValueError):
    """Raised when a benchmark suite configuration is invalid."""


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _expand_config_env(value, location="config"):
    """Recursively expand ${VAR} references and fail on missing variables."""
    import re

    if isinstance(value, dict):
        return {key: _expand_config_env(item, f"{location}.{key}") for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_config_env(item, f"{location}[{index}]") for index, item in enumerate(value)]
    if not isinstance(value, str):
        return value

    def replace(match):
        name = match.group(1)
        if name not in os.environ:
            raise BenchmarkConfigError(f"{location} references missing environment variable {name}")
        return os.environ[name]

    return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", replace, value)


def load_suite_config(path: str) -> Dict[str, Any]:
    """Load and validate the structural portion of a benchmark suite JSON file."""
    try:
        with open(path, "r", encoding="utf-8") as file:
            config = json.load(file)
    except FileNotFoundError as exc:
        raise BenchmarkConfigError(f"configuration file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise BenchmarkConfigError(
            f"invalid JSON in {path} at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc

    config = _expand_config_env(config)
    if not isinstance(config, dict):
        raise BenchmarkConfigError("configuration root must be a JSON object")
    if config.get("version") != SUITE_SCHEMA_VERSION:
        raise BenchmarkConfigError(
            f"version must be {SUITE_SCHEMA_VERSION} (got: {config.get('version')!r})"
        )

    allowed_root = {
        "version", "name", "description", "defaults", "cases", "report",
        "continue_on_error", "metadata",
    }
    unknown_root = sorted(set(config) - allowed_root)
    if unknown_root:
        raise BenchmarkConfigError(f"unknown top-level fields: {', '.join(unknown_root)}")
    if not isinstance(config.get("name", "benchmark-suite"), str):
        raise BenchmarkConfigError("name must be a string")
    if not isinstance(config.get("defaults", {}), dict):
        raise BenchmarkConfigError("defaults must be an object")
    if not isinstance(config.get("report", {}), dict):
        raise BenchmarkConfigError("report must be an object")
    if not isinstance(config.get("continue_on_error", True), bool):
        raise BenchmarkConfigError("continue_on_error must be true or false")

    cases = config.get("cases")
    if not isinstance(cases, list) or not cases:
        raise BenchmarkConfigError("cases must be a non-empty array")

    allowed_case = {
        "name", "description", "enabled", "tags", "preset", "scenario",
        "params", "matrix", "repeat",
    }
    seen_names = set()
    for index, case in enumerate(cases):
        location = f"cases[{index}]"
        if not isinstance(case, dict):
            raise BenchmarkConfigError(f"{location} must be an object")
        unknown = sorted(set(case) - allowed_case)
        if unknown:
            raise BenchmarkConfigError(f"{location} has unknown fields: {', '.join(unknown)}")
        name = case.get("name")
        if not isinstance(name, str) or not name.strip():
            raise BenchmarkConfigError(f"{location}.name must be a non-empty string")
        if name in seen_names:
            raise BenchmarkConfigError(f"duplicate case name: {name}")
        seen_names.add(name)
        if not isinstance(case.get("enabled", True), bool):
            raise BenchmarkConfigError(f"{location}.enabled must be true or false")
        if not isinstance(case.get("params", {}), dict):
            raise BenchmarkConfigError(f"{location}.params must be an object")
        if not isinstance(case.get("matrix", {}), dict):
            raise BenchmarkConfigError(f"{location}.matrix must be an object")
        if not isinstance(case.get("repeat", 1), int) or case.get("repeat", 1) < 1:
            raise BenchmarkConfigError(f"{location}.repeat must be a positive integer")
        tags = case.get("tags", [])
        if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
            raise BenchmarkConfigError(f"{location}.tags must be an array of strings")
        preset = case.get("preset")
        if preset is not None and preset not in get_preset_names():
            raise BenchmarkConfigError(
                f"{location}.preset must be one of: {', '.join(get_preset_names())}"
            )
        scenario = case.get("scenario")
        if scenario is not None and scenario not in SUITE_SCENARIOS:
            raise BenchmarkConfigError(
                f"{location}.scenario must be one of: {', '.join(sorted(SUITE_SCENARIOS))}"
            )
        for key, values in case.get("matrix", {}).items():
            if not isinstance(values, list) or not values:
                raise BenchmarkConfigError(f"{location}.matrix.{key} must be a non-empty array")

    report = config.get("report", {})
    unknown_report = sorted(set(report) - {"path", "include_request_details", "indent"})
    if unknown_report:
        raise BenchmarkConfigError(f"report has unknown fields: {', '.join(unknown_report)}")
    if not isinstance(report.get("include_request_details", False), bool):
        raise BenchmarkConfigError("report.include_request_details must be true or false")
    report_path = report.get("path")
    if report_path is not None and (not isinstance(report_path, str) or not report_path.strip()):
        raise BenchmarkConfigError("report.path must be a non-empty string when set")
    indent = report.get("indent", 2)
    if not isinstance(indent, int) or isinstance(indent, bool) or indent < 0:
        raise BenchmarkConfigError("report.indent must be a non-negative integer")
    return config


def _structured_workload_to_string(value, location: str) -> str:
    if isinstance(value, str):
        parse_workload_mix(value)
        return value
    if not isinstance(value, list) or not value:
        raise BenchmarkConfigError(f"{location} must be a workload string or non-empty array")
    parts = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise BenchmarkConfigError(f"{location}[{index}] must be an object")
        allowed = {"input_tokens", "output_tokens", "weight"}
        unknown = sorted(set(item) - allowed)
        if unknown:
            raise BenchmarkConfigError(
                f"{location}[{index}] has unknown fields: {', '.join(unknown)}"
            )
        try:
            input_tokens = int(item["input_tokens"])
            output_tokens = int(item["output_tokens"])
            weight = float(item["weight"])
        except (KeyError, TypeError, ValueError) as exc:
            raise BenchmarkConfigError(
                f"{location}[{index}] requires numeric input_tokens, output_tokens, and weight"
            ) from exc
        parts.append(f"{input_tokens}:{output_tokens}:{weight}")
    mix = ",".join(parts)
    parse_workload_mix(mix)
    return mix


def normalize_case_params(params: Dict[str, Any], location="params") -> Dict[str, Any]:
    """Flatten ergonomic nested case parameters into argparse destination names."""
    if not isinstance(params, dict):
        raise BenchmarkConfigError(f"{location} must be an object")

    aliases = {
        "input_tokens": "context_len",
        "output_tokens": "max_tokens",
        "request_count": "num_prompts",
        "num_requests": "num_prompts",
        "output_len": "max_tokens",
    }
    nested_maps = {
        "tokens": {
            "input": "context_len", "output": "max_tokens",
        },
        "requests": {
            "concurrency": "concurrency", "count": "num_prompts",
        },
        "prefix": {
            "shared": "share_prefix", "ratio": "prefix_ratio",
            "randomize": "random_shared_prefix", "warmup_rounds": "warmup_rounds",
        },
        "slo": {
            "ttft_seconds": "slo_ttft", "tpot_seconds": "slo_tpot",
        },
        "sweep": {
            "max_concurrency": "sweep_max_concurrency",
            "requests_per_round": "sweep_requests_per_round",
        },
        "slo_capacity": {
            "strategy": "slo_capacity_search_strategy",
            "linear_step": "slo_capacity_linear_step",
            "confirm_window": "slo_capacity_confirm_window",
            "confirm_rounds": "slo_capacity_confirm_rounds",
        },
        "pd": {
            "avg_input_tokens": "avg_input_tokens",
            "avg_output_tokens": "avg_output_tokens",
            "total_gpus": "total_gpus",
        },
    }

    normalized = {}

    def put(key, value, source):
        key = aliases.get(key.replace("-", "_"), key.replace("-", "_"))
        if key in normalized:
            raise BenchmarkConfigError(f"{location} sets {key} more than once ({source})")
        normalized[key] = value

    for raw_key, value in params.items():
        key = raw_key.replace("-", "_")
        if key in nested_maps:
            if not isinstance(value, dict):
                raise BenchmarkConfigError(f"{location}.{raw_key} must be an object")
            unknown = sorted(set(value) - set(nested_maps[key]))
            if unknown:
                raise BenchmarkConfigError(
                    f"{location}.{raw_key} has unknown fields: {', '.join(unknown)}"
                )
            for nested_key, nested_value in value.items():
                put(nested_maps[key][nested_key], nested_value, f"{raw_key}.{nested_key}")
        elif key == "warmup":
            if not isinstance(value, dict):
                raise BenchmarkConfigError(f"{location}.warmup must be an object")
            unknown = sorted(set(value) - {"enabled", "rounds", "requests_per_round"})
            if unknown:
                raise BenchmarkConfigError(
                    f"{location}.warmup has unknown fields: {', '.join(unknown)}"
                )
            if "enabled" in value:
                if not isinstance(value["enabled"], bool):
                    raise BenchmarkConfigError(f"{location}.warmup.enabled must be true or false")
                put("no_warmup", not value["enabled"], "warmup.enabled")
            if "rounds" in value:
                put("warmup_rounds", value["rounds"], "warmup.rounds")
            if "requests_per_round" in value:
                put(
                    "warmup_requests_per_round",
                    value["requests_per_round"],
                    "warmup.requests_per_round",
                )
        else:
            put(key, value, raw_key)

    if "api_key_env" in normalized:
        env_name = normalized.pop("api_key_env")
        if "api_key" in normalized:
            raise BenchmarkConfigError(f"{location} cannot set both api_key and api_key_env")
        if not isinstance(env_name, str) or not env_name:
            raise BenchmarkConfigError(f"{location}.api_key_env must be a non-empty string")
        if env_name not in os.environ:
            raise BenchmarkConfigError(
                f"{location}.api_key_env references missing environment variable {env_name}"
            )
        normalized["api_key"] = os.environ[env_name]

    if "workload_mix" in normalized:
        normalized["workload_mix"] = _structured_workload_to_string(
            normalized["workload_mix"], f"{location}.workload_mix"
        )
    return normalized


def _validate_api_key_transport(args, location: str) -> None:
    """Reject bearer credentials over plaintext non-loopback transports by default."""
    if args.mode != "api" or not args.api_key or args.allow_insecure_api_key:
        return
    try:
        parsed = urlparse(args.api_base)
        hostname = parsed.hostname.lower() if parsed.hostname else None
    except ValueError as exc:
        raise BenchmarkConfigError(f"{location}.api_base is not a valid URL: {exc}") from exc
    is_loopback = hostname in {"localhost", "127.0.0.1", "::1"}
    if parsed.scheme != "https" and not (parsed.scheme == "http" and is_loopback):
        raise BenchmarkConfigError(
            f"{location}: refusing to send api_key to non-HTTPS remote endpoint "
            f"{args.api_base!r}; use HTTPS or explicitly set allow_insecure_api_key=true"
        )


def _validate_effective_args(args, scenario: str, location: str) -> None:
    positive_ints = [
        "context_len", "max_tokens", "concurrency", "warmup_rounds",
        "sweep_max_concurrency", "avg_input_tokens", "avg_output_tokens",
    ]
    optional_positive_ints = [
        "num_prompts", "total_gpus", "sweep_requests_per_round",
        "warmup_requests_per_round",
    ]
    for key in positive_ints:
        value = getattr(args, key, None)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise BenchmarkConfigError(f"{location}.{key} must be a positive integer")
    for key in optional_positive_ints:
        value = getattr(args, key, None)
        if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 1):
            raise BenchmarkConfigError(f"{location}.{key} must be a positive integer when set")
    if args.slo_capacity_search_strategy not in {"linear", "binary-confirm"}:
        raise BenchmarkConfigError(
            f"{location}.slo_capacity_search_strategy must be linear or binary-confirm"
        )
    if (
        not isinstance(args.slo_capacity_linear_step, int)
        or isinstance(args.slo_capacity_linear_step, bool)
        or args.slo_capacity_linear_step < 1
    ):
        raise BenchmarkConfigError(
            f"{location}.slo_capacity_linear_step must be a positive integer"
        )
    if (
        not isinstance(args.slo_capacity_confirm_window, int)
        or isinstance(args.slo_capacity_confirm_window, bool)
        or args.slo_capacity_confirm_window < 0
    ):
        raise BenchmarkConfigError(
            f"{location}.slo_capacity_confirm_window must be a non-negative integer"
        )
    if (
        not isinstance(args.slo_capacity_confirm_rounds, int)
        or isinstance(args.slo_capacity_confirm_rounds, bool)
        or args.slo_capacity_confirm_rounds < 1
    ):
        raise BenchmarkConfigError(
            f"{location}.slo_capacity_confirm_rounds must be a positive integer"
        )
    if args.seed is not None and (not isinstance(args.seed, int) or isinstance(args.seed, bool)):
        raise BenchmarkConfigError(f"{location}.seed must be an integer when set")
    if args.dataset not in {"text", "random"}:
        raise BenchmarkConfigError(f"{location}.dataset must be text or random")
    for key in ("random_output_len",):
        value = getattr(args, key)
        if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 1):
            raise BenchmarkConfigError(f"{location}.{key} must be a positive integer when set")
    value = getattr(args, "random_input_len")
    if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
        raise BenchmarkConfigError(
            f"{location}.random_input_len must be a non-negative integer when set"
        )
    if (
        not isinstance(args.random_prefix_len, int)
        or isinstance(args.random_prefix_len, bool)
        or args.random_prefix_len < 0
    ):
        raise BenchmarkConfigError(f"{location}.random_prefix_len must be a non-negative integer")
    if args.random_seed is not None and (
        not isinstance(args.random_seed, int) or isinstance(args.random_seed, bool)
    ):
        raise BenchmarkConfigError(f"{location}.random_seed must be an integer when set")
    try:
        parse_range_ratio(args.random_range_ratio)
    except (TypeError, ValueError) as exc:
        raise BenchmarkConfigError(
            f"{location}.random_range_ratio must be R or INPUT_R,OUTPUT_R, with values in [0, 1]"
        ) from exc
    for key in [
        "share_prefix", "random_shared_prefix", "no_warmup", "ignore_eos",
        "allow_insecure_api_key",
    ]:
        if not isinstance(getattr(args, key), bool):
            raise BenchmarkConfigError(f"{location}.{key} must be true or false")
    if args.mode not in {"api", "offline"}:
        raise BenchmarkConfigError(f"{location}.mode must be api or offline")
    if not isinstance(args.model, str) or not args.model:
        raise BenchmarkConfigError(f"{location}.model must be a non-empty string")
    if args.tokenizer is not None and (not isinstance(args.tokenizer, str) or not args.tokenizer):
        raise BenchmarkConfigError(f"{location}.tokenizer must be null or a non-empty string")
    if not isinstance(args.prefix_ratio, (int, float)) or isinstance(args.prefix_ratio, bool) or not 0.0 <= args.prefix_ratio <= 1.0:
        raise BenchmarkConfigError(f"{location}.prefix_ratio must be between 0 and 1")
    if not isinstance(args.slo_ttft, (int, float)) or isinstance(args.slo_ttft, bool) or args.slo_ttft <= 0:
        raise BenchmarkConfigError(f"{location}.slo_ttft must be positive")
    if not isinstance(args.slo_tpot, (int, float)) or isinstance(args.slo_tpot, bool) or args.slo_tpot <= 0:
        raise BenchmarkConfigError(f"{location}.slo_tpot must be positive")
    if (
        not isinstance(args.max_failure_rate, (int, float))
        or isinstance(args.max_failure_rate, bool)
        or not 0.0 <= args.max_failure_rate <= 1.0
    ):
        raise BenchmarkConfigError(f"{location}.max_failure_rate must be between 0 and 1")
    if (
        not isinstance(args.min_goodput_pct, (int, float))
        or isinstance(args.min_goodput_pct, bool)
        or not 0.0 <= args.min_goodput_pct <= 100.0
    ):
        raise BenchmarkConfigError(f"{location}.min_goodput_pct must be between 0 and 100")
    if not isinstance(args.tp_size, int) or isinstance(args.tp_size, bool) or args.tp_size < 1:
        raise BenchmarkConfigError(f"{location}.tp_size must be a positive integer")
    if not isinstance(args.gpu_mem_util, (int, float)) or isinstance(args.gpu_mem_util, bool) or not 0 < args.gpu_mem_util <= 1:
        raise BenchmarkConfigError(f"{location}.gpu_mem_util must be in (0, 1]")
    if args.mode == "api" and (not isinstance(args.api_base, str) or not args.api_base):
        raise BenchmarkConfigError(f"{location}.api_base must be a non-empty string")
    if args.api_key is not None and (not isinstance(args.api_key, str) or not args.api_key):
        raise BenchmarkConfigError(f"{location}.api_key must be null or a non-empty string")
    _validate_api_key_transport(args, location)
    if args.mode == "offline" and scenario != "single":
        raise BenchmarkConfigError(f"{location}: {scenario} only supports mode=api")
    if scenario == "mixed-workload" and not args.workload_mix:
        raise BenchmarkConfigError(f"{location}: mixed-workload requires workload_mix")


def _build_case_args(defaults, case, matrix_values):
    parser = build_parser()
    args = parser.parse_args([])
    args.preset = case.get("preset")
    apply_preset(args, verbose=False)

    merged = {}
    for source, location in [
        (defaults, "defaults"),
        (case.get("params", {}), f"case {case['name']}.params"),
        (matrix_values, f"case {case['name']}.matrix"),
    ]:
        normalized = normalize_case_params(source, location)
        merged.update(normalized)

    allowed_params = set(vars(args)) - SUITE_MANAGEMENT_ARGS
    unknown = sorted(set(merged) - allowed_params)
    if unknown:
        raise BenchmarkConfigError(
            f"case {case['name']} has unknown benchmark parameters: {', '.join(unknown)}"
        )
    for key, value in merged.items():
        setattr(args, key, value)

    scenario = case.get("scenario")
    if scenario is None:
        if getattr(args, "_slo_capacity_search", False):
            scenario = "slo-capacity-search"
        elif getattr(args, "_pd_ratio", False):
            scenario = "pd-ratio"
        elif getattr(args, "_mixed_workload", False):
            scenario = "mixed-workload"
        elif getattr(args, "_sweep", False):
            scenario = "sweep"
        else:
            scenario = "single"
    else:
        args._slo_capacity_search = scenario == "slo-capacity-search"
        args._pd_ratio = scenario == "pd-ratio"
        args._mixed_workload = scenario == "mixed-workload"
        args._sweep = scenario == "sweep"

    apply_final_defaults(args)
    _validate_effective_args(args, scenario, f"case {case['name']}")
    return args, scenario


def expand_suite_cases(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Expand matrix dimensions and repeat counts into deterministic case runs."""
    import itertools

    expanded = []
    sequence = 0
    for case in config["cases"]:
        matrix = case.get("matrix", {})
        keys = list(matrix)
        combinations = itertools.product(*(matrix[key] for key in keys)) if keys else [()]
        for combination in combinations:
            matrix_values = dict(zip(keys, combination))
            suffix = ",".join(f"{key}={value}" for key, value in matrix_values.items())
            variant_name = case["name"] + (f"[{suffix}]" if suffix else "")
            for repeat_index in range(1, case.get("repeat", 1) + 1):
                sequence += 1
                display_name = variant_name
                if case.get("repeat", 1) > 1:
                    display_name += f"#repeat-{repeat_index}"
                expanded.append({
                    "id": f"case-{sequence:04d}",
                    "name": display_name,
                    "base_name": case["name"],
                    "description": case.get("description", ""),
                    "tags": case.get("tags", []),
                    "enabled": case.get("enabled", True),
                    "preset": case.get("preset"),
                    "scenario": case.get("scenario"),
                    "params": case.get("params", {}),
                    "matrix": matrix_values,
                    "repeat_index": repeat_index,
                })
    return expanded


def execute_benchmark(args, scenario: str):
    """Execute one normalized benchmark scenario and return serializable results."""
    if args.mode == "api":
        require_requests()
    if args.seed is not None:
        random.seed(args.seed)
    if scenario == "slo-capacity-search":
        return run_slo_capacity_benchmark(args)
    if scenario == "pd-ratio":
        return run_pd_ratio_benchmark(args)
    if scenario == "mixed-workload":
        return run_mixed_benchmark(args)
    if scenario == "sweep":
        return run_sweep_benchmark(args)
    if args.mode == "offline":
        return run_offline_benchmark(args)
    return run_api_benchmark(args)


def _quality_metric_samples(result: Dict[str, Any]):
    """Yield (label, metrics) pairs that represent request-bearing benchmark rounds."""
    if result.get("scenario") == "slo-capacity-search":
        selected_metrics = result.get("selected_metrics")
        if isinstance(selected_metrics, dict):
            yield "selected_capacity", selected_metrics
        return

    metrics = result.get("metrics")
    if isinstance(metrics, dict):
        yield "metrics", metrics

    for phase in ("prefill", "decode"):
        phase_metrics = result.get(phase)
        if isinstance(phase_metrics, dict):
            yield phase, phase_metrics

    history = result.get("history")
    if isinstance(history, list):
        for index, round_metrics in enumerate(history, start=1):
            if isinstance(round_metrics, dict):
                yield f"history[{index}]", round_metrics


def _case_has_request_failures(record: Dict[str, Any]) -> bool:
    """Return whether a completed case contains any failed API requests."""
    result = record.get("result")
    if not isinstance(result, dict):
        return False

    for _, metrics in _quality_metric_samples(result):
        failed = metrics.get("failed")
        if _is_finite_number(failed) and failed > 0:
            return True
        failure_rate = metrics.get("failure_rate")
        if not _is_finite_number(failure_rate):
            failure_rate = metrics.get("failed_ratio")
        if _is_finite_number(failure_rate) and failure_rate > 0:
            return True
    return False


def _quality_failures(result: Dict[str, Any], args) -> List[str]:
    """Return quality-gate violations without discarding the measured result."""
    if (
        result.get("scenario") == "slo-capacity-search"
        and not isinstance(result.get("selected_metrics"), dict)
    ):
        return [
            "slo-capacity-search found no concurrency satisfying the configured "
            "SLO pass-rate and failure-rate thresholds"
        ]

    violations = []
    for label, metrics in _quality_metric_samples(result):
        failure_rate = metrics.get("failure_rate")
        if not _is_finite_number(failure_rate):
            failure_rate = metrics.get("failed_ratio")
        if not _is_finite_number(failure_rate):
            if args.max_failure_rate < 1.0:
                violations.append(
                    f"{label}.failure_rate is unavailable; cannot verify "
                    f"max_failure_rate={args.max_failure_rate:.4f}"
                )
        elif failure_rate > args.max_failure_rate:
            violations.append(
                f"{label}.failure_rate={failure_rate:.4f} exceeds "
                f"max_failure_rate={args.max_failure_rate:.4f}"
            )

        if args.min_goodput_pct > 0:
            goodput_pct = metrics.get("goodput_pct")
            if not _is_finite_number(goodput_pct):
                violations.append(
                    f"{label}.goodput_pct is unavailable; cannot verify "
                    f"min_goodput_pct={args.min_goodput_pct:.2f}"
                )
            elif goodput_pct < args.min_goodput_pct:
                violations.append(
                    f"{label}.goodput_pct={goodput_pct:.2f} is below "
                    f"min_goodput_pct={args.min_goodput_pct:.2f}"
                )
    return violations


def _safe_effective_params(args) -> Dict[str, Any]:
    params = {}
    for key, value in vars(args).items():
        if key.startswith("_") or key in SUITE_MANAGEMENT_ARGS or key == "api_key":
            continue
        params[key] = value
    params["api_key_configured"] = bool(getattr(args, "api_key", None))
    return params


def _strip_request_details(value):
    if isinstance(value, dict):
        return {
            key: _strip_request_details(item)
            for key, item in value.items()
            if key != "results"
        }
    if isinstance(value, list):
        return [_strip_request_details(item) for item in value]
    return value


def _acquire_suite_checkpoint_lock(report_path: str):
    """Acquire a process-lifetime exclusive lock for a suite checkpoint."""
    lock_path = f"{os.path.abspath(report_path)}.lock"
    parent = os.path.dirname(lock_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    lock_file = open(lock_path, "a+", encoding="utf-8")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_file.close()
        raise BenchmarkConfigError(
            f"checkpoint {report_path!r} is already in use by another benchmark process"
        ) from exc
    return lock_file


def _release_suite_checkpoint_lock(lock_file) -> None:
    """Release a suite checkpoint lock while retaining its stable lock inode."""
    if lock_file is None:
        return
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    finally:
        lock_file.close()


def _write_json_report(path: str, report: Dict[str, Any], indent: int = 2) -> None:
    report_path = os.path.abspath(path)
    parent = os.path.dirname(report_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    temporary = f"{report_path}.tmp-{os.getpid()}"
    try:
        with open(temporary, "w", encoding="utf-8") as file:
            json.dump(
                report, file, ensure_ascii=False, indent=indent,
                sort_keys=False, allow_nan=False,
            )
            file.write("\n")
        os.replace(temporary, report_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _checkpoint_hash(value: Any) -> str:
    """Return a stable digest for checkpoint compatibility checks."""
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _checkpoint_case_key(case: Dict[str, Any], args, scenario: str) -> str:
    """Identify one expanded execution independently of its display position."""
    return _checkpoint_hash({
        "base_name": case["base_name"],
        "name": case["name"],
        "scenario": scenario,
        "repeat_index": case["repeat_index"],
        "matrix": case["matrix"],
        "enabled": case["enabled"],
        "params": _safe_effective_params(args),
    })


def _checkpoint_plan_fingerprint(prepared) -> str:
    """Fingerprint the selected, fully resolved suite execution plan."""
    return _checkpoint_hash([
        {"case_key": case_key, "id": case["id"]}
        for case, _, _, case_key in prepared
    ])


def _new_case_record(case: Dict[str, Any], args, scenario: str, case_key: str) -> Dict[str, Any]:
    """Create a checkpoint record before its case has started."""
    return {
        "id": case["id"],
        "case_key": case_key,
        "name": case["name"],
        "base_name": case["base_name"],
        "description": case["description"],
        "tags": case["tags"],
        "scenario": scenario,
        "repeat_index": case["repeat_index"],
        "matrix": case["matrix"],
        "status": "pending",
        "attempt": 0,
        "started_at": None,
        "finished_at": None,
        "duration_seconds": None,
        "params": _safe_effective_params(args),
        "result": None,
        "quality_failures": [],
        "error": None,
    }


def _load_resume_report(path: str, suite_name: str, plan_fingerprint: str) -> Dict[str, Any]:
    """Load a compatible suite checkpoint or explain how to start fresh."""
    try:
        with open(path, "r", encoding="utf-8") as file:
            report = json.load(file)
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkConfigError(
            f"cannot resume from report {path!r}: {exc}; use --no-resume to start fresh"
        ) from exc

    suite = report.get("suite")
    if not isinstance(suite, dict) or suite.get("name") != suite_name:
        raise BenchmarkConfigError(
            f"report {path!r} belongs to a different suite; use --no-resume to replace it"
        )
    if suite.get("execution_plan_sha256") != plan_fingerprint:
        raise BenchmarkConfigError(
            f"report {path!r} does not match the selected execution plan; "
            "use --no-resume to start fresh"
        )
    if not isinstance(report.get("cases"), list):
        raise BenchmarkConfigError(
            f"report {path!r} has no case checkpoint data; use --no-resume to start fresh"
        )
    return report


def _new_report(suite_name: str, config_path=None) -> Dict[str, Any]:
    return {
        "schema_version": SUITE_SCHEMA_VERSION,
        "suite": {
            "name": suite_name,
            "config_file": os.path.abspath(config_path) if config_path else None,
            "started_at": _now_iso(),
            "finished_at": None,
            "duration_seconds": None,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "hostname": platform.node(),
            "benchmark_script": os.path.abspath(__file__),
        },
        "summary": {
            "total": 0, "passed": 0, "failed": 0, "skipped": 0,
        },
        "cases": [],
    }


def _update_report_summary(report, started_perf, *, terminal: bool = False):
    cases = report["cases"]
    report["summary"] = {
        "total": len(cases),
        "pending": sum(case["status"] == "pending" for case in cases),
        "running": sum(case["status"] == "running" for case in cases),
        "passed": sum(case["status"] == "passed" for case in cases),
        "failed": sum(case["status"] == "failed" for case in cases),
        "interrupted": sum(case["status"] == "interrupted" for case in cases),
        "skipped": sum(case["status"] == "skipped" for case in cases),
    }
    if terminal:
        report["suite"]["finished_at"] = _now_iso()
        report["suite"]["duration_seconds"] = time.perf_counter() - started_perf
    else:
        report["suite"]["finished_at"] = None
        report["suite"]["duration_seconds"] = None


def _print_failed_case_summary(report: Dict[str, Any]) -> None:
    """Render a concise terminal summary for every failed suite case."""
    failed_cases = [
        case for case in report.get("cases", [])
        if case.get("status") == "failed"
    ]
    if not failed_cases:
        return

    print("\n失败用例详情:", file=sys.stderr)
    for case in failed_cases:
        error = case.get("error") if isinstance(case.get("error"), dict) else {}
        error_type = error.get("type", "UnknownError")
        message = error.get("message") or "; ".join(case.get("quality_failures", [])) or "未提供失败原因"
        print(
            f"  failed: {case.get('id', 'unknown')} {case.get('name', 'unknown')}\n"
            f"    reason: [{error_type}] {message}",
            file=sys.stderr,
        )


def _case_selected(case, case_filters, tag_filters):
    import fnmatch

    if case_filters and not any(
        fnmatch.fnmatch(case["name"], pattern) or fnmatch.fnmatch(case["base_name"], pattern)
        for pattern in case_filters
    ):
        return False
    if tag_filters and not any(tag in case["tags"] for tag in tag_filters):
        return False
    return True


def run_configured_suite(config_path: str, cli_args) -> int:
    config = load_suite_config(config_path)
    expanded = expand_suite_cases(config)

    selected = [
        case for case in expanded
        if _case_selected(case, cli_args.case_filters, cli_args.tag_filters)
    ]
    if not selected:
        raise BenchmarkConfigError("no benchmark cases matched the supplied filters")

    # Build every selected namespace during validation so errors fail before traffic starts.
    prepared = []
    for case in selected:
        args, scenario = _build_case_args(config.get("defaults", {}), case, case["matrix"])
        prepared.append((case, args, scenario, _checkpoint_case_key(case, args, scenario)))

    if cli_args.list_cases:
        for case, _, scenario, _ in prepared:
            state = "enabled" if case["enabled"] else "disabled"
            print(f"{case['id']}  {case['name']}  scenario={scenario}  {state}  tags={','.join(case['tags'])}")
        return 0
    if cli_args.validate_config:
        print(f"配置有效: {config_path} ({len(prepared)} 个展开后的测试用例)")
        return 0

    report_options = config.get("report", {})
    report_path = cli_args.report or report_options.get("path")
    # Microseconds plus PID prevent colliding report names when multiple
    # benchmark processes start in the same second.
    timestamp = f"{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}-p{os.getpid()}"
    if not report_path:
        report_path = f"benchmark-report-{timestamp}.json"
    else:
        # A report path containing {timestamp} always starts a new checkpoint.
        report_path = report_path.replace("{timestamp}", timestamp)
    if not os.path.isabs(report_path):
        report_path = os.path.join(os.path.dirname(os.path.abspath(config_path)), report_path)
    include_details = report_options.get("include_request_details", False)
    indent = report_options.get("indent", 2)
    continue_on_error = config.get("continue_on_error", True) and not cli_args.fail_fast
    plan_fingerprint = _checkpoint_plan_fingerprint(prepared)

    checkpoint_lock = _acquire_suite_checkpoint_lock(report_path)
    try:
        report = None
        if not cli_args.no_resume:
            report = _load_resume_report(
                report_path, config.get("name", "benchmark-suite"), plan_fingerprint
            )
        resumed = report is not None
        if report is None:
            report = _new_report(config.get("name", "benchmark-suite"), config_path)
    except Exception:
        _release_suite_checkpoint_lock(checkpoint_lock)
        raise

    now = _now_iso()
    if resumed:
        report["suite"].setdefault("first_started_at", report["suite"].get("started_at"))
        report["suite"]["resumed_at"] = now
    report["suite"].update({
        "name": config.get("name", "benchmark-suite"),
        "config_file": os.path.abspath(config_path),
        "started_at": now,
        "description": config.get("description", ""),
        "metadata": config.get("metadata", {}),
        "execution_plan_sha256": plan_fingerprint,
        "run_state": "running",
        "updated_at": _now_iso(),
    })
    report["environment"] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hostname": platform.node(),
        "benchmark_script": os.path.abspath(__file__),
    }

    previous_records = {
        record.get("case_key"): record
        for record in report.get("cases", [])
        if isinstance(record, dict) and isinstance(record.get("case_key"), str)
    }
    records = []
    retry_request_failure_count = 0
    for case, args, scenario, case_key in prepared:
        previous = previous_records.get(case_key)
        if (
            previous
            and previous.get("status") == "passed"
            and not _case_has_request_failures(previous)
        ):
            record = previous
        else:
            if previous and previous.get("status") == "passed":
                retry_request_failure_count += 1
            record = _new_case_record(case, args, scenario, case_key)
            if previous:
                record["attempt"] = previous.get("attempt", 0)
        if not case["enabled"] and record.get("status") != "passed":
            record["status"] = "skipped"
            record["skip_reason"] = "disabled"
        records.append(record)
    report["cases"] = records

    started_perf = time.perf_counter()
    # Write the complete pending plan before traffic starts. A later hard stop
    # can therefore never make a not-yet-started case look completed.
    _update_report_summary(report, started_perf)
    try:
        _write_json_report(report_path, report, indent)
    except KeyboardInterrupt:
        report["suite"]["run_state"] = "interrupted"
        report["suite"]["updated_at"] = _now_iso()
        _update_report_summary(report, started_perf, terminal=True)
        _write_json_report(report_path, report, indent)
        _release_suite_checkpoint_lock(checkpoint_lock)
        raise
    except (OSError, TypeError, ValueError) as exc:
        _release_suite_checkpoint_lock(checkpoint_lock)
        raise BenchmarkConfigError(f"cannot write report to {report_path!r}: {exc}") from exc

    print(f"Benchmark suite: {report['suite']['name']}")
    print(f"Cases: {len(prepared)}")
    print(f"Report: {report_path}")
    if resumed:
        skipped_passed = sum(record["status"] == "passed" for record in records)
        print(f"Resume: skipping {skipped_passed} previously passed case(s)")
        if retry_request_failure_count:
            print(
                "Resume: retrying "
                f"{retry_request_failure_count} previously passed case(s) "
                "with request-level failures"
            )

    abort_remaining = False
    interrupted = False
    for position, ((case, args, scenario, _), record) in enumerate(zip(prepared, records), start=1):
        if record["status"] == "passed":
            print(f"[{position}/{len(prepared)}] {case['name']} (passed; resume skip)")
            continue
        if not case["enabled"]:
            continue
        if abort_remaining:
            continue

        print()
        print("=" * 80)
        print(f"[{position}/{len(prepared)}] {case['name']} ({scenario})")
        print("=" * 80)
        case_started = time.perf_counter()
        try:
            record.update({
                "status": "running",
                "attempt": record.get("attempt", 0) + 1,
                "started_at": _now_iso(),
                "finished_at": None,
                "duration_seconds": None,
                "result": None,
                "quality_failures": [],
                "error": None,
            })
            report["suite"]["updated_at"] = _now_iso()
            _update_report_summary(report, started_perf)
            _write_json_report(report_path, report, indent)

            result = execute_benchmark(args, scenario)
            if result is None:
                raise RuntimeError("benchmark produced no successful result")
            record["result"] = result if include_details else _strip_request_details(result)
            record["quality_failures"] = _quality_failures(result, args)
            if record["quality_failures"]:
                record["status"] = "failed"
                record["error"] = {
                    "type": "QualityGateError",
                    "message": "; ".join(record["quality_failures"]),
                }
                print(
                    f"ERROR: case {case['name']} failed quality gates: "
                    f"{record['error']['message']}",
                    file=sys.stderr,
                )
                if not continue_on_error:
                    abort_remaining = True
            else:
                record["status"] = "passed"
        except KeyboardInterrupt:
            record["status"] = "interrupted"
            record["error"] = {"type": "KeyboardInterrupt", "message": "interrupted by user"}
            report["suite"]["run_state"] = "interrupted"
            report["suite"]["updated_at"] = _now_iso()
            abort_remaining = True
            interrupted = True
            raise
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = {"type": type(exc).__name__, "message": str(exc)}
            print(f"ERROR: case {case['name']} failed: {exc}", file=sys.stderr)
            if not continue_on_error:
                abort_remaining = True
        finally:
            record["finished_at"] = _now_iso()
            record["duration_seconds"] = time.perf_counter() - case_started
            report["suite"]["updated_at"] = _now_iso()
            _update_report_summary(report, started_perf, terminal=interrupted)
            _write_json_report(report_path, report, indent)
            if interrupted:
                _release_suite_checkpoint_lock(checkpoint_lock)

    try:
        report["suite"]["run_state"] = "completed" if not abort_remaining else "stopped"
        report["suite"]["updated_at"] = _now_iso()
        _update_report_summary(report, started_perf, terminal=True)
        _write_json_report(report_path, report, indent)
    except KeyboardInterrupt:
        report["suite"]["run_state"] = "interrupted"
        report["suite"]["updated_at"] = _now_iso()
        _update_report_summary(report, started_perf, terminal=True)
        _write_json_report(report_path, report, indent)
        _release_suite_checkpoint_lock(checkpoint_lock)
        raise
    exit_code = 1 if report["summary"]["failed"] else 0
    _release_suite_checkpoint_lock(checkpoint_lock)
    print()
    print(
        f"Suite complete: passed={report['summary']['passed']}, "
        f"failed={report['summary']['failed']}, "
        f"interrupted={report['summary']['interrupted']}, "
        f"pending={report['summary']['pending']}, "
        f"skipped={report['summary']['skipped']}"
    )
    _print_failed_case_summary(report)
    print(f"JSON report: {os.path.abspath(report_path)}")
    return exit_code


# ── CLI & Main ─────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with all CLI options."""
    parser = argparse.ArgumentParser(
        description="LLM 长上下文 Prefill / Decode 性能实测脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=build_preset_epilog(),
    )
    parser.add_argument(
        "--config", default=None,
        help="JSON benchmark suite 配置文件；启用后按 cases/matrix/repeat 执行并输出 JSON 报告"
    )
    parser.add_argument(
        "--report", default=None,
        help="JSON 报告输出路径。配置模式下覆盖 report.path；传统单次模式下启用结果输出"
    )
    parser.add_argument(
        "--case", action="append", default=[], dest="case_filters",
        help="配置模式下按用例名或 fnmatch 模式筛选，可重复指定"
    )
    parser.add_argument(
        "--tag", action="append", default=[], dest="tag_filters",
        help="配置模式下只运行包含指定 tag 的用例，可重复指定"
    )
    parser.add_argument(
        "--list-cases", action="store_true",
        help="展开并列出配置中的测试用例，不发送请求"
    )
    parser.add_argument(
        "--validate-config", action="store_true",
        help="校验配置、参数和 matrix，不发送请求"
    )
    parser.add_argument(
        "--fail-fast", action="store_true",
        help="配置模式下任一用例失败后跳过剩余用例"
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="配置模式下忽略同路径的历史报告并从头运行；默认会恢复，并只跳过没有请求级失败的已通过用例"
    )
    parser.add_argument(
        "--preset", choices=get_preset_names(), default=None,
        help="预设测试用例。选择后自动设置最优参数，仍可用其他选项覆盖"
    )
    parser.add_argument(
        "--mode", choices=["offline", "api"], default="offline",
        help="offline: 直接用vLLM LLM类测量(需本地GPU，最精确); "
             "api: 通过已启动的vllm serve服务测量TTFT(更贴近真实部署)"
    )
    parser.add_argument(
        "--model", default="Qwen/Qwen3.6-35B-A3B",
        help="[offline模式] 模型权重路径或HuggingFace repo id；"
             "[api模式] 请求体里的 model 字段，需与 vllm serve 启动时"
             "注册的模型名一致(可用 curl <api-base>/models 查询实际名称)"
    )
    parser.add_argument(
        "--tokenizer", default=None,
        help="单独指定 tokenizer 路径(本地目录或repo id)。"
             "api模式下 --model 只是服务端注册的名字，不一定能直接被"
             "AutoTokenizer 加载，这时用此参数单独指向本地权重/tokenizer目录，"
             "例如 --tokenizer /data/models/Qwen3.6-35B-A3B。"
             "不填则默认与 --model 相同"
    )
    parser.add_argument(
        "--dataset", choices=["text", "random"], default=None,
        help="测试数据集：text 使用原有中文填充文本；random 使用可复现的合成 token 序列"
    )
    parser.add_argument(
        "--random-input-len", type=int, default=None,
        help="[random数据集] 输入长度均值；默认使用 --context-len；可设为 0 表示仅使用共享前缀"
    )
    parser.add_argument(
        "--random-output-len", type=int, default=None,
        help="[random数据集] 输出长度均值；默认使用 --max-tokens"
    )
    parser.add_argument(
        "--random-prefix-len", type=int, default=0,
        help="[random数据集] 所有请求共享的合成 token 前缀长度（默认：0）"
    )
    parser.add_argument(
        "--random-range-ratio", default="0.0",
        help="[random数据集] 长度采样范围比例，可为 R 或 INPUT_R,OUTPUT_R（默认：0）"
    )
    parser.add_argument(
        "--random-seed", type=int, default=None,
        help="[random数据集] 独立 NumPy RNG 的种子；未设置时回退到 --seed"
    )
    parser.add_argument(
        "--context-len", type=int, default=None,
        help="测试的上下文长度(token数)，默认128K=131072"
    )
    parser.add_argument(
        "--concurrency", "-c", type=int, default=None,
        help="并发请求数 (默认: 1)"
    )
    parser.add_argument(
        "--num-prompts", "--num-requests", "-n", type=int, default=None, dest="num_prompts",
        help="测试发出的总请求数。若未指定，默认与 --concurrency (并发数) 相同"
    )
    parser.add_argument(
        "--share-prefix", action="store_true", default=None,
        help="并发请求是否共享相同 Prompt 前缀。开启时按 --prefix-ratio 共享前缀；"
             "不开启(默认)时每个请求注入独立随机前缀(强制 Cache Miss，实测纯算力 Prefill)"
    )
    parser.add_argument(
        "--random-shared-prefix", action="store_true",
        help="配合 --share-prefix 使用。开启后，每次运行脚本时的共享前缀会加上随机UUID"
             "（保证本次运行所有请求共享该前缀，但跨次运行必定 Cache Miss）。"
    )
    parser.add_argument(
        "--prefix-ratio", type=float, default=1.0,
        help="共享前缀比例 (0.0 ~ 1.0，默认 1.0 代表 100%% 前缀共享)。仅在开启 --share-prefix 时生效"
    )
    parser.add_argument(
        "--max-tokens", "--output-len", type=int, default=None, dest="max_tokens",
        help="控制输出生成 Token 长度 (默认: 1)"
    )
    parser.add_argument(
        "--tp-size", type=int, default=2,
        help="[offline模式] tensor parallel size，对应GPU数量"
    )
    parser.add_argument(
        "--gpu-mem-util", type=float, default=0.90,
        help="[offline模式] GPU显存利用率上限(0~1)"
    )
    parser.add_argument(
        "--api-base", default="http://localhost:8000/v1",
        help="[api模式] API 服务地址，可以是自建 vllm serve 地址，"
             "也可以是云端服务地址(如 Alibaba Cloud Model Studio、OpenRouter 等)"
    )
    parser.add_argument(
        "--api-key", default=None,
        help="[api模式] 云端服务通常需要鉴权，传入API Key "
             "(也可以设置环境变量 API_KEY 后用 --api-key $API_KEY 传入)"
    )
    parser.add_argument(
        "--allow-insecure-api-key", action="store_true",
        help="允许向非 HTTPS 的远端 API 发送 API Key（默认拒绝；loopback HTTP 不受限）"
    )
    parser.add_argument(
        "--no-warmup", action="store_true",
        help="跳过预热阶段 (默认会先进行预热以排除初始化与 CUDA Graph 捕获开销)"
    )
    parser.add_argument(
        "--warmup-rounds", type=int, default=1, dest="warmup_rounds",
        help="预热轮数。sweep/SLO 使用当前正式 workload 在扫描前预热指定轮数；"
             "默认每轮仅 1 条请求，可通过 --warmup-requests-per-round 覆盖（默认: 1）"
    )
    parser.add_argument(
        "--warmup-requests-per-round", type=int, default=None,
        dest="warmup_requests_per_round",
        help="[sweep/SLO] 每轮正式 workload 预热发送的请求数；默认 1，"
             "不会沿用正式扫描的 max(2×并发, 4) 请求数规则"
    )
    parser.add_argument(
        "--ignore-eos", action="store_true", default=True,
        help="强制忽略 EOS 终止符，确保生成到指定的 max_tokens 长度 (默认开启)"
    )
    parser.add_argument(
        "--no-ignore-eos", action="store_false", dest="ignore_eos",
        help="允许模型在遇到 EOS 终止符时提前结束"
    )
    parser.add_argument(
        "--sweep-max-concurrency", type=int, default=128,
        help="[sweep模式] 最大搜索并发数 (默认: 128)"
    )
    parser.add_argument(
        "--sweep-requests-per-round", type=int, default=None,
        help="[sweep模式] 每个并发档位的请求数；默认自动取 max(2×并发, 4)"
    )
    parser.add_argument(
        "--slo-capacity-search-strategy", choices=["linear", "binary-confirm"],
        default="binary-confirm",
        help="[slo-capacity-sweep] 精扫策略：linear 逐并发扫描至首次新失败，"
             "再扫描后续确认窗口；binary-confirm 二分定位后确认候选窗口并重复验证（默认）"
    )
    parser.add_argument(
        "--slo-capacity-linear-step", type=int, default=1,
        help="[slo-capacity-sweep] linear 精扫的并发采样步长（默认: 1，逐并发）；"
             "首次失败后的确认窗口始终逐并发扫描"
    )
    parser.add_argument(
        "--slo-capacity-confirm-window", type=int, default=8,
        help="[slo-capacity-sweep] linear 首次新失败后的后续确认档位数；"
             "binary-confirm 候选两侧的逐并发确认窗口（默认: 8）"
    )
    parser.add_argument(
        "--slo-capacity-confirm-rounds", type=int, default=3,
        help="[slo-capacity-sweep] linear 和 binary-confirm 对最终候选要求的总测量轮数（默认: 3）"
    )
    parser.add_argument(
        "--avg-input-tokens", type=int, default=4096, dest="avg_input_tokens",
        help="[pd-ratio模式] 业务平均输入 token 数 (默认: 4096)"
    )
    parser.add_argument(
        "--avg-output-tokens", type=int, default=2048, dest="avg_output_tokens",
        help="[pd-ratio模式] 业务平均输出 token 数 (默认: 2048)"
    )
    parser.add_argument(
        "--total-gpus", type=int, default=None, dest="total_gpus",
        help="[pd-ratio模式] 可用 GPU 总数，用于生成实际部署建议"
    )
    parser.add_argument(
        "--workload-mix", type=str, default=None, dest="workload_mix",
        help="[mixed-workload模式] 混合负载定义。"
             "格式: 'input:output:weight,input:output:weight,...'  "
             "例如: '128:4096:0.3,8192:1024:0.5,131072:1:0.2' 表示 "
             "30%% 短输入长输出 + 50%% 中等 + 20%% 长输入短输出"
    )
    parser.add_argument(
        "--slo-ttft", type=float, default=5.0,
        help="SLO 阈值: 允许的最大 TTFT (秒)，默认 5.0"
    )
    parser.add_argument(
        "--slo-tpot", type=float, default=0.1,
        help="SLO 阈值: 允许的最大 TPOT (秒)，默认 0.1 (100ms)"
    )
    parser.add_argument(
        "--max-failure-rate", type=float, default=0.0,
        help="质量门禁: 允许的最大请求失败率，范围 0~1（默认 0，不允许请求失败）"
    )
    parser.add_argument(
        "--min-goodput-pct", type=float, default=0.0,
        help="最低 SLO Goodput 百分比（0~100）。slo-capacity-sweep 将其用作每轮通过/停止阈值；"
             "设为 0 时容量搜索保持严格 100%% 通过率，其他场景禁用此质量门禁（默认：0）"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="随机种子，用于复现混合负载分配；默认不固定"
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.config:
        try:
            return run_configured_suite(args.config, args)
        except BenchmarkConfigError as exc:
            print(f"ERROR: invalid benchmark configuration: {exc}", file=sys.stderr)
            return 2
    if args.list_cases or args.validate_config or args.case_filters or args.tag_filters or args.fail_fast or args.no_resume:
        parser.error("--list-cases/--validate-config/--case/--tag/--fail-fast/--no-resume require --config")

    apply_preset(args)
    apply_final_defaults(args)

    if args._slo_capacity_search:
        scenario = "slo-capacity-search"
    elif args._pd_ratio:
        scenario = "pd-ratio"
    elif args._mixed_workload or getattr(args, "workload_mix", None):
        scenario = "mixed-workload"
    elif args._sweep:
        scenario = "sweep"
    else:
        scenario = "single"
    started_perf = time.perf_counter()
    started_at = _now_iso()
    record = {
        "id": "case-0001",
        "name": args.preset or scenario,
        "base_name": args.preset or scenario,
        "description": "legacy command-line benchmark",
        "tags": [],
        "scenario": scenario,
        "repeat_index": 1,
        "matrix": {},
        "status": "failed",
        "started_at": started_at,
        "finished_at": None,
        "duration_seconds": None,
        "params": _safe_effective_params(args),
        "result": None,
        "quality_failures": [],
        "error": None,
    }
    report = None
    if args.report:
        report = _new_report("single-benchmark")
        report["suite"]["started_at"] = started_at
        report["cases"].append(record)
        try:
            _write_json_report(args.report, report)
        except (OSError, TypeError, ValueError) as exc:
            print(f"ERROR: cannot write report to {args.report!r}: {exc}", file=sys.stderr)
            return 2

    try:
        _validate_effective_args(args, scenario, "command line")
    except BenchmarkConfigError as exc:
        if report is None:
            parser.error(str(exc))
        record["error"] = {"type": type(exc).__name__, "message": str(exc)}
        record["finished_at"] = _now_iso()
        record["duration_seconds"] = time.perf_counter() - started_perf
        _update_report_summary(report, started_perf)
        _write_json_report(args.report, report)
        print(f"ERROR: invalid benchmark arguments: {exc}", file=sys.stderr)
        print(f"JSON report: {os.path.abspath(args.report)}")
        return 2

    if NSYS_PROFILE:
        print()
        print("  ── Nsys Profiling ──")
        print(f"    Container: {NSYS_CONTAINER}")
        print(f"    Session:   {NSYS_SESSION}")
        print("    nsys start/stop will be triggered automatically around the benchmark round")
        print()

    exit_code = 1
    try:
        if args.mode == "api":
            require_requests()
            collect_and_print_system_info(args)
        result = execute_benchmark(args, scenario)
        if result is None:
            raise RuntimeError("benchmark produced no successful result")
        record["result"] = _strip_request_details(result)
        record["quality_failures"] = _quality_failures(result, args)
        if record["quality_failures"]:
            record["error"] = {
                "type": "QualityGateError",
                "message": "; ".join(record["quality_failures"]),
            }
            print(f"ERROR: benchmark failed quality gates: {record['error']['message']}", file=sys.stderr)
        else:
            record["status"] = "passed"
            exit_code = 0
    except KeyboardInterrupt:
        record["error"] = {"type": "KeyboardInterrupt", "message": "interrupted by user"}
        raise
    except Exception as exc:
        record["error"] = {"type": type(exc).__name__, "message": str(exc)}
        print(f"ERROR: benchmark failed: {exc}", file=sys.stderr)
    finally:
        record["finished_at"] = _now_iso()
        record["duration_seconds"] = time.perf_counter() - started_perf
        if report is not None:
            _update_report_summary(report, started_perf)
            _write_json_report(args.report, report)
            print(f"JSON report: {os.path.abspath(args.report)}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())