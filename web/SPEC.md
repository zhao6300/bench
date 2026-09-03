# Web 管理 / 结果查看 UI — 设计规范

> 本文档描述~LLM 推理 benchmark 工具的 Web 管理与结果查看面板。文档语言为中文。

## 1. 目标

为 `benchmark/benchmark.py` 提供一个独立于终端的 Web 管理与结果浏览 UI，满足以下能力：

1. **实时监控**：打开浏览器即可看到正在运行的基准任务当前用例、轮次进度、请求成功/失败计数。
2. **运行管理**：通过一个表单提交新的 benchmark 运行，支持从内置示例加载配置、编辑 JSON、提交前校验，并对正在运行的任务执行终止。
3. **结果浏览**：浏览本地已经生成的 JSON 报告文件（包括 `benchmark-report*.json` 和 `web/runs` 目录下的内容）。
4. **非侵入**：面板托盘进程完全独立于测量路径，崩溃或关闭面板不影响基准结果。
5. **零前端/后端依赖**：Node.js 使用内置 `http` 模块，无 npm 依赖；浏览器端使用原生 HTML/CSS/JS。

## 2. 非目标

- 不实现多用户/鉴权体系；安全模型是「本机回环 + 可选环境变量 bearer token」。
- 不修改 Python 端的 benchmark 逻辑、报告格式或 CLI 参数。
- 不引入 React/Vue 等前端框架；UI 以原生 DOM API + 自定义 CSS 实现。
- 不实现复杂的实时数据可视化（除进度条与简单纯文本指标外不引入图表库）。

## 3. 架构

```
┌──────────────┐        Lifecycle events (POST /ingest)      ┌───────────────┐
│  benchmark   │ ───────────────────────────────────────────► │  Node.js 面板  │
│  Python 进程 │ ◄──── LLM_BENCHMARK_WEB_TOKEN="<run_id>"    │  server.js    │
└──────────────┘                                             └───────────────┘
                                                              │ SSE push        │
                                                              ▼                 │
                                                     ┌───────────────┐         │
                                                     │  浏览器 SPA    │◄────────┘
                                                     │  (原生 HTML)   │
                                                     └───────────────┘
```

- **Click-through**：Panel 进程和 benchmark 进程是父子进程关系；panel 通过 `child_process.spawn` 启动 benchmark，并通过 stdio 捕获 stdout/stderr。benchmark 进程通过 best-effort HTTP POST 把生命周期事件发送到 panel 的 `/ingest`。
- **SSE 分发**：浏览器通过 GET `/events` 与 panel 建立 SSE 连接。panel 把新收到的 ingest 事件立刻推送给所有订阅者。
- **Run 路由**：panel 在 spawn 子进程时通过环境变量注入 `LLM_BENCHMARK_WEB_TOKEN=<run_id>`；子进程每次推送都携带 `Authorization: Bearer <run_id>`，panel 按 token 把事件路由到正确的 run，从而在同一个 panel 里同时看到多次运行的实时状态（虽然通常只有一个活跃运行）。

## 4. HTTP API Spec

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET /` | | 服务前端 `index.html` |
| `GET /app.js` / `GET /style.css` | | 服务前端静态文件 |
| `POST /ingest` | | benchmark → panel 的 lifecycle 事件推送入口 |
| `GET /events` | | 浏览器 SSE 订阅入口 |
| `GET /api/status` | | 当前面板一次性快照 |
| `GET /api/runs` | | 全部 runs 列表（内存 + 磁盘去重） |
| `GET /api/runs/:id` | | 单个 run 详情 |
| `POST /api/runs` | | 提交新运行，body `{config: {...}}` |
| `DELETE /api/runs/:id` | | 销毁或干净地终止运行 |
| `GET /api/examples` | | 列出 `examples/*.json` 内置配置 |
| `GET /api/reports` | | 列出本地 JSON 报告文件 |
| `GET /api/reports/:file` | | 读取报告 JSON 文件 |
| `POST /api/reset` | | 清空面板内存状态 |

### 4.1 `POST /ingest`

复用现有 Python 端协议。请求 body 为 JSON，字段包括：

```json
{
  "event": "case_started",
  "case_name": "smoke",
  "scenario": "single",
  "position": 1,
  "total_cases": 4,
  "message": "case 1/4 开始: smoke (single)"
}
```

Header 可选携带 `Authorization: Bearer <token>`。panel 用 token 查找对应 run；找不到事件回退到全局 ring buffer，但 run-scoped 字段（case_position、round 等）不会更新。

### 4.2 `GET /api/status` 响应

```json
{
  "active_run_id": "r20260902-1",
  "active_run": { ...RunSummary },
  "events": [ ...最近 100 条事件 ],
  "total_runs": 3
}
```

### 4.3 `RunSummary`

```json
{
  "id": "...",
  "name": "...",
  "status": "running",
  "case_position": 2,
  "case_total": 6,
  "current_case": {"name":"decode-throughput","scenario":"single"},
  "stage": "warmup",
  "round": {"total_requests": 24, "concurrency": 4, "completed": 18, "succeeded": 17, "failed": 1},
  "started_at": "...",
  "finished_at": null
}
```

### 4.4 Run 生命周期状态机

```
  queued ──► running ──► passed
                      └──► failed
                      └──► interrupted
  queued ──► failed  (spawn 错误)
  说明：从 "queued" 到 "running" 只会在子进程真正 spawn 成功后发生；
  spawn 失败会直接进入 "failed"。任何终态都会通过 POST /api/reset 全部清空。
```

### 4.5 `POST /api/runs`

用户若是通过 UI 提交，body 长这样：

```json
{"config":  {"version":1, "name":"vllm-ci", "cases":[ ... ]}}
```

校验失败返回 `400 {error}`；成功返回 `201 {id, status: "queued"}`。panel 会把 config 写入 `web/runs/<id>-config.json` 之后用 `.venv/bin/python` 启动子进程。

### 4.6 `GET /api/reports`

扫描两个目录：

1. **仓库根目录**（`process.cwd()` 下，匹配 `benchmark-report*.json`、`*-report-*.json`）
2. **web/runs/**（panel 子进程产出的报告）

返回按 `mtime` 倒序的列表。

## 5. 前端信息架构

### 5.1 布局

```
┌─────────────────────────────────────────────────────────────┐
│  🟠 LLM Benchmark   [ 仪表盘 ][ 压测任务 ][ 结果报告 ]   🌓 │ ← 顶栏 sticky
├─────────────────────────────────────────────────────────────┤
│ ↑ 内容区 max-width 1200                                     │
│                                                             │
│ ─ 仪表盘 tab（默认）：                                       │
│   [总任务数][运行中][已完成][失败] ← KPI card grid           │
│   [ 当前运行卡片（进度条 / 当前用例 / 阶段 / 状态） ]          │
│   [ 最近事件 feed ][ 环境信息卡 ]  ← 两栏                    │
│                                                             │
│ ─ 压测任务 tab：                                             │
│   [ 新建压测 ] 按钮，右上角                                   │
│   [ 表格：任务ID / 名称 / 状态 / 进度 / 开始 / 操作 ]          │
│   [ 任务详情卡（阶段 tabs：日志 / 事件 / 报告） ]              │
│                                                             │
│ ─ 结果报告 tab：                                             │
│   [ 报告卡片 grid（文件名、大小、mtime） ]                    │
│   [ Modal 查看单份报告：套件信息 / 用例统计 / 结果表 ]          │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 Material Design 3 tokens

选中 Google 色板，用 CSS 变量维护，方便后续微调：

```css
--m-primary: #1a73e8;         /* Google Blue */
--m-primary-hover: #1557b0;
--success: #188038;           /* Google Green */
--danger: #d93025;            /* Google Red */
--warn: #ef6c00;              /* Google Orange */
```

形状：
- **卡片**：圆角 16px（`--radius-lg`），1px 边框（`--bg-outline`），hover 时加 `--shadow-1`。
- **按钮**：圆角 100px（pill）或 4px；primary、text 三档。
- **状态 pill**：圆角 100px，背景为对应颜色的 10% 透明度版本。
- **表格**：单元格 12px padding，1px 分隔线。

**阴影**：Material 的 elevation 1（`--shadow-1`）用于悬浮卡片；elevation 2（`--shadow-2`）用于模态框。

字体：Roboto（fallback 系统字体），等宽字体用 `Roboto Mono` / `SF Mono`，数字用 `font-variant-numeric: tabular-nums` 防抖动。

### 5.3 主题切换

- 支持亮色与暗色，亮色为默认。
- 右上角图标按钮切换；偏好保存到 `localStorage`。
- 暗色主题使用 `--bg-root: #131314`、`--bg-surface: #1e1f20`，与 `--text-primary: #e8eaed` 的 Google Dark 调色板对齐。

## 6. 前端行为

### 6.1 路由

使用 `#` fragment 客户端路由，不刷新页面：

```
#dashboard  #runs  #reports
```

切换 tab 时：
- 仪表盘：open SSE，加载快照
- 任务：调用 `GET /api/runs`
- 报告：调用 `GET /api/reports`

### 6.2 实时更新

SSE 接收的事件：

| SSE event | 处理 |
|-----------|------|
| `snapshot` | 用全量状态渲染 KPI、活跃运行、事件 feed |
| `ingest` | 把新事件 prepend 到最近事件 feed；若当前查看对应 run 则刷新 run detail |
| `run_update` | 状态变化，重新加载 runs 列表或 run detail |
| `log` | 刷新当前打开 run detail 的日志 |

### 6.3 空状态

- 仪表盘 KPI 全 0；活跃运行卡片隐藏；事件 feed 显示占位说明。
- 任务表显示「尚无压测任务，点击右上角新建压测提交」。
- 报告 grid 显示「暂无基准报告 JSON 文件」。

### 6.4 提交新压测

步骤：
1. 点击「新建压测」→ 打开 dialog。
2. Dialog 内顶部展示 exemple chips（从 `/api/examples` 拉），点击 chip 即把配套 JSON 填入 textarea。
3. 或直接在 textarea 粘贴 JSON。
4. **校验**：「校验配置」使用前端 JSON parse 校验，确认含 `cases` 数组；错误降级到前端提示。
5. **提交**：POST `/api/runs`，成功关闭 dialog 并跳到 runs tab 选中新任务。

### 6.5 终止 / 删除运行

表格每行末尾有「详情」和「终止」按钮。终止单次行为是 `DELETE /api/runs/:id`：
1. 若子进程还在运行，先 `SIGTERM`，3 秒后未退出则升级到 `SIGKILL`。
2. 从 runs Map 中移除该 run。
3. 通过 SSE 通知其他浏览器更新列表。

## 7. 兼容性

- 复用 `benchmark/web_management.py` 已有的 `WebPushProgressReporter`，不改 Python 端行为。
- 兼容现有 `--web-port` / `--web-host` 参数。
- `web/runs/*` 是新增的目录，由 panel 生成，不进版本库。
- 保持 panel 完全可选：关闭时一切照旧，只有 `POST /ingest` 端点存在。

## 8. 安全

- 只对 `LLM_BENCHMARK_WEB_TOKEN` 注入的 bear token 做等值校验；不匹配返回 `403`，最佳努力策略。
- 静态文件服务限制在 `web/public/` 下，避免任意文件读取。
- 若 `LLM_BENCHMARK_WEB_HOST` 不是回环地址，打印告警。
- 子进程输出行「不直接回显到浏览器」（无 XSS 可能），都走 `textContent` 或 `esc()`。

## 9. 测试策略

- 使用 Node.js 内置 `node:test` 测试框架（零外部依赖），测试以下行为：
  1. `newRunId` 唯一且格式正确
  2. `applyIngest` 在 `case_started` / `round_started` / `request_finished` / `final_results` 下的状态机迁移
  3. `findRunByToken` 路由
  4. `listReports` / `isReportName` 的 glob 语义与 traversal 拒绝
  5. `listExamples` 是否返回 JSON 格式
  6. 提交新 run 时 config 校验错误
- 前端不做自动化测试（避免额外依赖），通过手动浏览器验证 UI 行为。
- 端到端 idle smoke test：启动 `node web/server.js` 后 `curl` 验证 `/api/status`、`/api/runs`、`/api/examples`、`/api/reports` 的响应码。

## 10. 实现范围

本 PR 提供：
- `web/server.js`：Node 后端，含 parser/routing/子进程管理/SSE。
- `web/public/index.html` / `style.css` / `app.js`：原生 SPA。
- `web/package.json`：npm 元数据。
- `web/SPEC.md`：本文。
- README 更新：说明如何启动 panel 及与 CLI 的关系。

**不做**：
- 不修改 `benchmark/benchmark.py`。
- 不引入 npm 依赖。
- 不改动现有 Python 测试断言。
