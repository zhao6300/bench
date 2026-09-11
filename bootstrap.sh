#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ui_root="$repo_root/web/ui"
python="$repo_root/.venv/bin/python"
serve_mode=0

usage() {
  cat <<'EOF'
Usage: bootstrap.sh [--serve COMMAND...]

Options:
  --help, -h          Show this help.
  --serve COMMAND...  Run a command with the benchmark Python after setup.

Without --serve, the script only installs dependencies.
EOF
}

if [[ $# -gt 0 ]]; then
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --serve)
      serve_mode=1
      shift
      if [[ $# -eq 0 ]]; then
        echo "--serve 需要一个命令与可选参数" >&2
        exit 2
      fi
      ;;
    *)
      echo "未知参数: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
fi

if [[ ! -x "$python" ]]; then
  echo "缺少 $python。请先运行 uv venv ." >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "缺少 uv。请先安装 uv。" >&2
  exit 1
fi

if ! command -v pnpm >/dev/null 2>&1; then
  echo "缺少 pnpm。请先安装 pnpm。" >&2
  exit 1
fi

cd "$repo_root"

echo "安装 benchmark 与 lint 依赖..."
uv pip install --python "$python" -r requirements/common.txt
uv pip install --python "$python" -r requirements/lint.txt
uv pip install --python "$python" -e .
"$python" -m pre_commit install

echo "安装 Web UI 依赖..."
cd "$ui_root"
pnpm install --frozen-lockfile
cd "$repo_root"

if [[ "$serve_mode" -eq 0 ]]; then
  echo "依赖安装完成。"
  exit 0
fi

exec "$@"
