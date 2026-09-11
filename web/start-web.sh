#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ui_root="$repo_root/web/ui"
python="$repo_root/.venv/bin/python"
host="127.0.0.1"
port="8000"

usage() {
  cat <<'EOF'
Usage: web/start-web.sh [--host HOST] [--port PORT]

Arguments:
  --host HOST     Bind host. Defaults to 127.0.0.1; only loopback is allowed.
  --port PORT     Bind port. Defaults to 8000.
  -h, --help      Show this help.

说明：若 dist/index.html 已存在且前端源码或配置没有更新，则跳过构建。
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      [[ $# -ge 2 ]] || { echo "--host 需要一个值" >&2; exit 2; }
      host="$2"
      shift 2
      ;;
    --port)
      [[ $# -ge 2 ]] || { echo "--port 需要一个值" >&2; exit 2; }
      port="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "未知参数: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -x "$python" ]]; then
  echo "缺少 $python。请先在仓库根目录初始化 .venv。" >&2
  exit 1
fi

if ! command -v pnpm >/dev/null 2>&1; then
  echo "缺少 pnpm。请先安装 pnpm。" >&2
  exit 1
fi

if [[ ! -d "$ui_root/node_modules" ]]; then
  echo "安装前端依赖..."
  (
    cd "$ui_root"
    if [[ -f pnpm-lock.yaml ]]; then
      pnpm install --frozen-lockfile
    else
      pnpm install
    fi
  )
fi

if [[ ! -f "$ui_root/dist/index.html" ]] || {
  find "$ui_root/src" "$ui_root/index.html" "$ui_root/tsconfig.json" "$ui_root/vite.config.ts" "$ui_root/package.json" "$ui_root/pnpm-lock.yaml" -type f -newer "$ui_root/dist/index.html" -print -quit | grep -q .
}; then
  echo "检测到前端需要重建。"
  (cd "$ui_root" && pnpm build)
else
  echo "前端产物已是最新，跳过构建。"
fi

echo "启动报告服务: http://$host:$port/"
cd "$repo_root"
exec "$python" -m web.serve --host "$host" --port "$port" --print-address
