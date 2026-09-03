/* LLM Benchmark Dashboard — frontend logic (vanilla ES5, zero dependencies). */

(function () {
  'use strict';

  var $ = function (id) { return document.getElementById(id); };

  var els = {
    conn: $('info-conn'),
    kpi_total: $('kpi-total'),
    kpi_running: $('kpi-running'),
    kpi_passed: $('kpi-passed'),
    kpi_failed: $('kpi-failed'),
    active_card: $('active-card'),
    active_status: $('active-status-pill'),
    active_name: $('active-name'),
    active_id: $('active-id'),
    active_progress: $('active-progress'),
    active_case: $('active-case'),
    active_round: $('active-round'),
    active_stage: $('active-stage'),
    event_feed: $('event-feed'),
    event_count: $('event-count'),
    info_conn: $('info-conn'),
    info_active: $('info-active'),
    info_total: $('info-total'),
    info_url: $('info-url'),
    runs_body: $('runs-body'),
    run_detail: $('run-detail'),
    detail_name: $('detail-name'),
    detail_id: $('detail-id'),
    detail_status: $('detail-status-pill'),
    detail_kvs: $('detail-kvs'),
    detail_log_view: $('detail-log-view'),
    detail_events_feed: $('detail-events-feed'),
    detail_report_box: $('detail-report-box'),
    panel_logs: $('panel-logs'),
    panel_events: $('panel-events'),
    panel_report: $('panel-report'),
    btn_new_run: $('btn-new-run'),
    btn_back_runs: $('btn-back-runs'),
    btn_view_active: $('btn-view-active'),
    btn_theme: $('btn-theme'),
    reports_grid: $('reports-grid'),
    report_modal_overlay: $('report-modal-overlay'),
    report_modal_title: $('modal-title'),
    report_modal_body: $('modal-body'),
    btn_close_modal: $('btn-close-modal'),
    example_chips: $('example-chips'),
    config_textarea: $('config-textarea'),
    btn_validate: $('btn-validate-config'),
    btn_launch: $('btn-launch'),
    submit_msg: $('submit-message'),
    submit_view: $('view-submit'),
    view_dashboard: $('view-dashboard'),
    view_runs: $('view-runs'),
    view_reports: $('view-reports'),
    toast: $('toast'),
    runs_list_card: $('runs-list'),
  };

  var currentTab = 'dashboard';
  var selectedRunId = null;
  var currentReportSummaryFn = null;

  /* ── Helpers ─────────────────────────────────────────────────── */

  function esc(s) {
    var div = document.createElement('div');
    div.textContent = s == null ? '' : String(s);
    return div.innerHTML;
  }

  function statusText(status) {
    var map = {
      queued: '排队中', running: '运行中', passed: '通过',
      failed: '失败', interrupted: '已中断', skipped: '已跳过', pending: '待运行',
    };
    return map[status] || status || '—';
  }

  function fmtTime(iso) {
    if (!iso) return '—';
    var d = new Date(iso);
    var pad = function (n) { return n < 10 ? '0' + n : String(n); };
    return pad(d.getHours()) + ':' + pad(d.getMinutes()) + ':' + pad(d.getSeconds());
  }

  function fmtDateTime(iso) {
    if (!iso) return '—';
    var d = new Date(iso);
    var pad = function (n) { return n < 10 ? '0' + n : String(n); };
    return d.getFullYear() + '-' + pad(d.getMonth() + 1) + '-' + pad(d.getDate()) + ' ' + pad(d.getHours()) + ':' + pad(d.getMinutes());
  }

  function fmtDuration(sec) {
    if (sec == null || isNaN(sec)) return '—';
    if (sec < 60) return sec.toFixed(1) + 's';
    var m = Math.floor(sec / 60);
    var s = Math.round(sec % 60);
    return m + 'm' + s + 's';
  }

  function fmtNum(v) {
    if (v == null || isNaN(v)) return '—';
    return Number(v).toLocaleString('zh-CN', {maximumFractionDigits: 1});
  }

  function showToast(msg) {
    els.toast.textContent = msg;
    els.toast.style.display = '';
    setTimeout(function () { els.toast.style.display = 'none'; }, 2200);
  }

  /* ── Tab navigation (hash routing) ──────────────────────────── */

  function showTab(tab) {
    currentTab = tab;
    els.view_dashboard.style.display = tab === 'dashboard' ? '' : 'none';
    els.view_runs.style.display = tab === 'runs' ? '' : 'none';
    els.view_reports.style.display = tab === 'reports' ? '' : 'none';
    els.submit_view.style.display = tab === 'submit' ? '' : 'none';
    document.querySelectorAll('.tab').forEach(function (el) {
      var isActive = el.dataset.route === tab;
      el.classList.toggle('active', isActive);
      if (el.setAttribute) el.setAttribute('aria-selected', String(isActive));
    });
    if (tab === 'dashboard') { refreshDashboard(); }
    if (tab === 'runs') { loadRuns(); }
    if (tab === 'reports') { loadReports(); }
    if (tab === 'submit') { initSubmitForm(); }
  }

  document.querySelectorAll('.tab').forEach(function (el) {
    el.addEventListener('click', function (e) {
      e.preventDefault();
      var tab = el.dataset.route || 'dashboard';
      if (location.hash !== '#' + tab) location.hash = tab;
      showTab(tab);
    });
  });

  window.addEventListener('hashchange', function () {
    var tab = (location.hash || '#dashboard').replace(/^#/, '');
    showTab(tab);
  });

  /* ── SSE connection ─────────────────────────────────────────── */

  var evtSource = null;
  var reconnectTimer = null;

  function connect() {
    if (evtSource) evtSource.close();
    evtSource = new EventSource('/events');
    evtSource.addEventListener('open', function () {
      els.info_conn.textContent = '已连接';
      els.info_conn.style.color = 'var(--success)';
    });
    evtSource.addEventListener('snapshot', function (e) {
      var snap = JSON.parse(e.data);
      renderSnapshot(snap);
      if (currentTab === 'runs' && !selectedRunId && snap.runs) {
        renderRunsTable(snap.runs);
      }
      if (currentTab === 'runs' && selectedRunId) {
        fetchRunDetailSoon(selectedRunId);
      }
    });
    evtSource.addEventListener('log', function () {
      if (selectedRunId) fetchRunDetailSoon(selectedRunId);
    });
    evtSource.addEventListener('error', function () {
      els.info_conn.textContent = '未连接';
      els.info_conn.style.color = 'var(--danger)';
      evtSource = null;
      if (!reconnectTimer) reconnectTimer = setTimeout(function () { reconnectTimer = null; connect(); }, 3000);
    });
  }

  var detailRefreshTimer = null;
  function fetchRunDetailSoon(runId) {
    if (detailRefreshTimer) return;
    detailRefreshTimer = setTimeout(function () {
      detailRefreshTimer = null;
      fetchRunDetail(runId);
    }, 150);
  }

  /* ── Dashboard rendering ────────────────────────────────────── */

  function refreshDashboard() {
    fetch('/api/status').then(function (r) { return r.json(); }).then(renderSnapshot).catch(function () {
      els.info_conn.textContent = '无法连接';
      els.info_conn.style.color = 'var(--danger)';
    });
  }

  function renderSnapshot(snap) {
    els.info_total.textContent = String(snap.total_runs || 0);
    els.kpi_total.textContent = String(snap.total_runs || 0);
    var list = snap.runs || [];
    var runningCount = list.filter(function (r) { return r.status === 'running' || r.status === 'queued'; }).length;
    var passedCount = list.filter(function (r) { return r.status === 'passed'; }).length;
    var failedCount = list.filter(function (r) { return r.status === 'failed' || r.status === 'interrupted'; }).length;
    els.kpi_running.textContent = String(runningCount);
    els.kpi_passed.textContent = String(passedCount);
    els.kpi_failed.textContent = String(failedCount);
    els.info_active.textContent = snap.active_run_id ? snap.active_run_id : '无';

    var active = snap.active_run;
    if (active) {
      els.active_card.style.display = '';
      els.active_status.textContent = statusText(active.status);
      els.active_status.className = 'status-pill ' + String(active.status).replace(/[^a-z0-9_-]/gi, '');
      els.active_name.textContent = active.name || active.id;
      els.active_id.textContent = active.id;
      var pct = active.case_total > 0 ? Math.round((active.case_position / active.case_total) * 100) : 0;
      els.active_progress.style.width = Math.min(pct, 100) + '%';
      els.active_case.textContent =
        active.current_case ? '当前: ' + active.current_case.name + ' (' + active.current_case.scenario + ')'
        : '用例进度 ' + (active.case_position || 0) + '/' + (active.case_total || 0);
      if (active.round && active.round.total_requests > 0) {
        els.active_round.textContent = '并发 ' + (active.round.concurrency || 0) + ' · ' +
          (active.round.completed || 0) + '/' + (active.round.total_requests || 0) +
          ' 成功 ' + (active.round.succeeded || 0) + ' 失败 ' + (active.round.failed || 0);
      } else {
        els.active_round.textContent = '无活跃轮次';
      }
      els.active_stage.textContent = active.stage ? '阶段: ' + active.stage : '等待阶段';
    } else {
      els.active_card.style.display = 'none';
    }

    // On first load, render the event feed from the snapshot.
    els.event_feed.innerHTML = '';
    els.event_count.textContent = '0';
    if (snap.events && snap.events.length) {
      for (var i = 0; i < snap.events.length; i += 1) {
        prependEvent(snap.events[i], false);
      }
      els.event_count.textContent = String(snap.events.length);
    } else if (!els.event_feed.querySelector('.event-row')) {
      els.event_feed.innerHTML = '<div class="empty-hint">尚未接收到生命周期事件。<br>启动 benchmark 时添加 <code>--web-port 3000</code>。</div>';
    }
  }

  function prependEvent(ev, increment) {
    if (els.event_feed.querySelector('.empty-hint')) {
      els.event_feed.innerHTML = '';
    }
    var item = document.createElement('div');
    item.className = 'event-row';
    var time = document.createElement('span');
    time.className = 'event-time';
    time.textContent = fmtTime(ev.receivedAt);
    var msg = document.createElement('span');
    msg.className = 'event-msg';
    msg.textContent = ev.message || ev.event || '';
    item.appendChild(time);
    item.appendChild(msg);
    els.event_feed.insertBefore(item, els.event_feed.firstChild);
    if (increment !== false) {
      els.event_count.textContent = String(parseInt(els.event_count.textContent || '0', 10) + 1);
    }
    while (els.event_feed.children.length > 60) {
      els.event_feed.removeChild(els.event_feed.lastChild);
    }
  }

  /* ── Runs table ─────────────────────────────────────────────── */

  function renderRunsTable(runsList) {
    if (!runsList.length) {
      els.runs_body.innerHTML = '<tr><td colspan="6" class="table-empty">尚无压测任务。点击右上角「新建压测」提交。</td></tr>';
      return;
    }
    var rows = runsList.map(function (run) {
      var status = String(run.status || 'queued').replace(/[^a-z0-9_-]/g, '');
      var progress = run.case_total > 0
        ? run.case_position + '/' + run.case_total
        : '—';
      return '<tr data-run-id="' + esc(run.id) + '">' +
        '<td class="mono" style="font-size:12px; max-width:140px; overflow:hidden; text-overflow:ellipsis;">' + esc(run.id) + '</td>' +
        '<td>' + esc(run.name) + '</td>' +
        '<td><span class="status-pill ' + status + '">' + esc(statusText(run.status)) + '</span></td>' +
        '<td class="mono">' + esc(progress) + '</td>' +
        '<td class="mono" style="font-size:12px">' + esc(run.started_at ? fmtDateTime(run.started_at) : '—') + '</td>' +
        '<td>' +
          '<button class="link-btn view-run">详情</button>' +
          '<button class="link-btn stop-run" style="color:var(--danger)">终止</button>' +
        '</td></tr>';
    }).join('');
    els.runs_body.innerHTML = rows;
    Array.prototype.forEach.call(els.runs_body.querySelectorAll('tr'), function (tr) {
      var id = tr.dataset.runId;
      var viewButtons = tr.querySelectorAll('.view-run');
      var stopButtons = tr.querySelectorAll('.stop-run');
      Array.prototype.forEach.call(viewButtons, function (b) {
        b.addEventListener('click', function () { viewRun(id); });
      });
      Array.prototype.forEach.call(stopButtons, function (b) {
        b.addEventListener('click', function () { deleteRun(id); });
      });
    });
  }

  function loadRuns() {
    fetch('/api/runs').then(function (r) { return r.json(); }).then(function (body) {
      renderRunsTable(body.runs || []);
    }).catch(function () {
      els.runs_body.innerHTML = '<tr><td colspan="6" class="table-empty">无法连接 API.</td></tr>';
    });
  }

  /* ── Run detail ─────────────────────────────────────────────── */

  function viewRun(runId) {
    selectedRunId = runId;
    els.runs_list_card.style.display = 'none';
    els.run_detail.style.display = '';
    fetchRunDetail(runId);
  }

  function hideRunDetail() {
    selectedRunId = null;
    els.runs_list_card.style.display = '';
    els.run_detail.style.display = 'none';
  }

  els.btn_back_runs.addEventListener('click', function () { hideRunDetail(); loadRuns(); });

  els.btn_view_active.addEventListener('click', function (e) {
    e.preventDefault();
    showTab('runs');
    var activeEl = document.getElementById('info-active');
    if (activeEl && activeEl.textContent && activeEl.textContent !== '无') {
      setTimeout(function () { viewRun(activeEl.textContent); }, 50);
    }
  });

  function fetchRunDetail(runId) {
    fetch('/api/runs/' + encodeURIComponent(runId)).then(function (r) {
      if (r.status === 404) { hideRunDetail(); throw new Error('removed'); }
      if (!r.ok) throw new Error(r.status);
      return r.json();
    }).then(function (run) {
      if (!run || run.id !== runId) return;
      els.detail_name.textContent = run.name || '—';
      els.detail_id.textContent = run.id;
      els.detail_status.textContent = statusText(run.status);
      els.detail_status.className = 'status-pill ' + String(run.status || '').replace(/[^a-z0-9_-]/g, '');

      var kvs = [
        ['状态', statusText(run.status)],
        ['开始时间', run.started_at ? fmtDateTime(run.started_at) : '—'],
        ['结束时间', run.finished_at ? fmtDateTime(run.finished_at) : '—'],
        ['退出码', run.exit_code == null ? '—' : String(run.exit_code)],
        ['进程 PID', run.pid == null ? '—' : String(run.pid)],
        ['用例进度', (run.case_position || 0) + '/' + (run.case_total || 0)],
      ];
      els.detail_kvs.innerHTML = kvs.map(function (p) {
        return '<div class="kv-item"><div class="kv-label">' + esc(p[0]) + '</div><div class="kv-value">' + esc(p[1]) + '</div></div>';
      }).join('');

      // Logs
      var logs = run.logs || [];
      els.detail_log_view.textContent = logs.map(function (l) {
        return '[' + l.stream + '] ' + l.line;
      }).join('\n') || '尚无日志输出';
      els.detail_log_view.scrollTop = els.detail_log_view.scrollHeight;

      // Events
      var events = run.events || [];
      if (events.length) {
        els.detail_events_feed.innerHTML = events.slice(-40).reverse().map(function (ev) {
          var tz = document.createElement('span');
          tz.textContent = ev.message || ev.event || '';
          var msg = tz.innerHTML;
          return '<div class="event-row">' +
            '<span class="event-time">' + fmtTime(ev.receivedAt) + '</span>' +
            '<span class="event-msg">' + msg + '</span></div>';
        }).join('');
      } else {
        els.detail_events_feed.innerHTML = '<div class="empty-hint">暂无生命周期事件。</div>';
      }

      // Report
      els.detail_report_box.innerHTML = run.final_report
        ? renderReportSummary(run.final_report)
        : '<div class="empty-hint">报告尚未生成。任务完成或失败后在此查看。</div>';
    }).catch(function () {
      // Errors other than 404 are likely network hiccups; keep last-rendered detail.
    });
  }

  document.querySelectorAll('.chip[data-panel]').forEach(function (chip) {
    chip.addEventListener('click', function () {
      document.querySelectorAll('.chip[data-panel]').forEach(function (c) { c.classList.remove('active'); });
      chip.classList.add('active');
      var p = chip.dataset.panel;
      els.panel_logs.style.display = p === 'logs' ? '' : 'none';
      els.panel_events.style.display = p === 'events' ? '' : 'none';
      els.panel_report.style.display = p === 'report' ? '' : 'none';
    });
  });

  function deleteRun(runId) {
    if (!window.confirm('终止并删除压测任务 ' + runId + '？')) return;
    fetch('/api/runs/' + encodeURIComponent(runId), {method: 'DELETE'}).then(function () {
      showToast('已删除 ' + runId);
      loadRuns();
    }).catch(function () {
      showToast('删除失败');
    });
  }

  /* ── Submit new run ─────────────────────────────────────────── */

  var examplesLoaded = false;
  var examples = [];

  function initSubmitForm() {
    if (!examplesLoaded) loadExamples();
  }

  function loadExamples() {
    fetch('/api/examples').then(function (r) { return r.json(); }).then(function (body) {
      examples = body.examples || [];
      if (!examples.length) {
        els.example_chips.innerHTML = '<span style="color: var(--text-hint); font-size: 13px;">未找到内置示例。</span>';
        return;
      }
      var html = examples.map(function (ex, i) {
        return '<button class="chip" data-idx="' + i + '">' + esc(ex.name) + ' (' + ex.cases + ' 用例)</button>';
      }).join('');
      els.example_chips.innerHTML = html;
      Array.prototype.forEach.call(els.example_chips.querySelectorAll('.chip'), function (chip) {
        chip.addEventListener('click', function () {
          Array.prototype.forEach.call(els.example_chips.querySelectorAll('.chip'), function (c) { c.classList.remove('active'); });
          chip.classList.add('active');
          var ex = examples[parseInt(chip.dataset.idx, 10)];
          els.config_textarea.value = JSON.stringify(ex.config, null, 2);
          els.submit_msg.textContent = '已加载 ' + ex.name;
          els.submit_msg.className = 'form-message ok';
        });
      });
      examplesLoaded = true;
    }).catch(function () {
      els.example_chips.innerHTML = '<span style="color: var(--danger); font-size: 12.5px;">无法加载示例。</span>';
    });
  }

  els.btn_validate.addEventListener('click', function () {
    var result = validateConfigText();
    els.submit_msg.className = 'form-message ' + (result.ok ? 'ok' : 'err');
    els.submit_msg.textContent = result.ok ? '✓ JSON 格式正确，含 ' + result.caseCount + ' 个用例' : '✗ ' + result.error;
  });

  function validateConfigText() {
    var text = els.config_textarea.value.trim();
    if (!text) return {ok: false, error: '配置为空'};
    var cfg;
    try {
      cfg = JSON.parse(text);
    } catch (err) {
      return {ok: false, error: 'JSON 无效: ' + err.message};
    }
    if (!cfg || typeof cfg !== 'object' || Array.isArray(cfg)) return {ok: false, error: '配置必须是 JSON 对象'};
    if (!Array.isArray(cfg.cases)) return {ok: false, error: '配置必须包含 cases 数组'};
    if (!cfg.cases.length) return {ok: false, error: 'cases 数组不能为空'};
    return {ok: true, caseCount: cfg.cases.length};
  }

  els.btn_launch.addEventListener('click', function () {
    var result = validateConfigText();
    if (!result.ok) {
      els.submit_msg.className = 'form-message err';
      els.submit_msg.textContent = '✗ ' + result.error;
      return;
    }
    var config = JSON.parse(els.config_textarea.value);
    fetch('/api/runs', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({config: config}),
    }).then(function (r) { return r.json().then(function (b) { return {ok: r.ok, body: b}; }); }).then(function (r) {
      if (!r.ok) {
        els.submit_msg.className = 'form-message err';
        els.submit_msg.textContent = '✗ ' + (r.body.error || '提交失败');
        return;
      }
      els.submit_msg.className = 'form-message ok';
      els.submit_msg.textContent = '✓ 已提交运行 ' + r.body.id;
      showToast('已启动压测 ' + r.body.id);
      setTimeout(function () {
        showTab('runs');
        setTimeout(function () { viewRun(r.body.id); }, 50);
      }, 300);
    }).catch(function (err) {
      els.submit_msg.className = 'form-message err';
      els.submit_msg.textContent = '✗ 请求失败: ' + err.message;
    });
  });

  /* ── Reports browse + modal ─────────────────────────────────── */

  function loadReports() {
    els.reports_grid.innerHTML = '<div class="empty-state">加载中…</div>';
    fetch('/api/reports').then(function (r) { return r.json(); }).then(function (body) {
      var list = body.reports || [];
      if (!list.length) {
        els.reports_grid.innerHTML = '<div class="empty-state">暂无基准报告 JSON 文件。运行一次压测后会在这里看到报告。</div>';
        return;
      }
      els.reports_grid.innerHTML = list.map(function (r) {
        return '<div class="report-card" data-file="' + esc(r.file) + '">' +
          '<div class="report-name">' + esc(r.file) + '</div>' +
          '<div class="report-meta">' +
            '<span>📦 ' + (r.size / 1024).toFixed(1) + ' KB</span>' +
            '<span>⏱ ' + fmtDateTime(r.modifiedAt) + '</span>' +
          '</div></div>';
      }).join('');
      Array.prototype.forEach.call(els.reports_grid.querySelectorAll('.report-card'), function (card) {
        card.addEventListener('click', function () { viewReport(card.dataset.file); });
      });
    }).catch(function () {
      els.reports_grid.innerHTML = '<div class="empty-state">无法连接报告 API.</div>';
    });
  }

  function viewReport(fileName) {
    els.report_modal_title.textContent = fileName;
    els.report_modal_body.innerHTML = '<div style="color: var(--text-hint); padding: 20px; text-align: center;">加载中…</div>';
    els.report_modal_overlay.style.display = 'flex';
    fetch('/api/reports/' + encodeURIComponent(fileName)).then(function (r) {
      if (!r.ok) throw new Error(r.status);
      return r.json();
    }).then(function (report) {
      els.report_modal_body.innerHTML = renderReportSummary(report);
    }).catch(function () {
      els.report_modal_body.innerHTML = '<div style="color: var(--danger); text-align: center; padding: 20px;">报告加载失败。</div>';
    });
  }

  els.btn_close_modal.addEventListener('click', function () { els.report_modal_overlay.style.display = 'none'; });
  els.report_modal_overlay.addEventListener('click', function (e) {
    if (e.target === els.report_modal_overlay) els.report_modal_overlay.style.display = 'none';
  });

  currentReportSummaryFn = function (report) { return renderReportSummary(report); };

  function renderReportSummary(report) {
    var suite = report.suite || {};
    var summary = report.summary || {};
    var env = report.environment || {};
    var html = '';
    html += '<div class="detail-section"><div class="detail-section-title">套件信息</div><div class="detail-kv-grid">';
    html += kvItem('名称', suite.name || '—');
    html += kvItem('状态', suite.run_state || '—');
    html += kvItem('开始时间', suite.started_at ? fmtDateTime(suite.started_at) : '—');
    html += kvItem('结束时间', suite.finished_at ? fmtDateTime(suite.finished_at) : '—');
    html += kvItem('总耗时', suite.duration_seconds != null ? fmtDuration(suite.duration_seconds) : '—');
    html += kvItem('主机', env.hostname || '—');
    html += '</div></div>';

    html += '<div class="detail-section"><div class="detail-section-title">用例统计</div><div class="detail-kv-grid">';
    html += kvItem('总数', summary.total != null ? summary.total : 0);
    html += kvItem('通过', summary.passed != null ? summary.passed : 0);
    html += kvItem('失败', summary.failed != null ? summary.failed : 0);
    html += kvItem('跳过', summary.skipped != null ? summary.skipped : 0);
    html += kvItem('中断', summary.interrupted != null ? summary.interrupted : 0);
    html += kvItem('待运行', summary.pending != null ? summary.pending : 0);
    html += '</div></div>';

    var cases = report.cases || [];
    if (cases.length) {
      html += '<div class="detail-section"><div class="detail-section-title">用例详情 (' + cases.length + ' 条)</div>';
      html += '<div class="metrics-table-wrap"><table class="metrics-table"><thead><tr>' +
        '<th>名称</th><th>场景</th><th>状态</th><th>耗时</th><th>吞吐</th><th>错误</th>' +
        '</tr></thead><tbody>';
      cases.forEach(function (c) {
        var status = String(c.status || '').replace(/[^a-z0-9_-]/g, '');
        var txt = document.createElement('span'); txt.textContent = c.status || '—';
        var metrics = (c.result && c.result.metrics) ? c.result.metrics : (c.metrics || {});
        var tput = metrics.overall_throughput || metrics.throughput || null;
        var errEl = document.createElement('span'); errEl.textContent = c.error || '—';
        html += '<tr>' +
          '<td>' + esc(c.name || '—') + '</td>' +
          '<td><span class="scenario-tag">' + esc(c.scenario || '—') + '</span></td>' +
          '<td><span class="case-badge ' + status + '">' + esc(statusText(c.status)) + '</span></td>' +
          '<td class="value mono">' + (c.duration_seconds != null ? fmtDuration(c.duration_seconds) : '—') + '</td>' +
          '<td class="value mono">' + (tput ? fmtNum(tput) + ' tok/s' : '—') + '</td>' +
          '<td class="value" style="max-width:200px; word-break: break-word;">' + errEl.innerHTML + '</td>' +
        '</tr>';
      });
      html += '</tbody></table></div></div>';
    }
    return html;
  }

  function kvItem(label, value) {
    return '<div class="detail-kv"><div class="detail-kv-label">' + esc(label) + '</div><div class="detail-kv-value">' + esc(String(value)) + '</div></div>';
  }

  /* ── Theme toggle ───────────────────────────────────────────── */

  els.btn_theme.addEventListener('click', function () {
    document.body.classList.toggle('dark-theme');
    var isDark = document.body.classList.contains('dark-theme');
    try { localStorage.setItem('llm-bench-theme', isDark ? 'dark' : 'light'); } catch (e) { /* ignore */ }
  });

  try {
    if (localStorage.getItem('llm-bench-theme') === 'dark') {
      document.body.classList.add('dark-theme');
    }
  } catch (e) { /* ignore */ }

  /* ── Init ───────────────────────────────────────────────────── */

  var initialHash = (window.location.hash || '#dashboard').replace(/^#/, '');
  if (['dashboard', 'runs', 'reports', 'submit'].indexOf(initialHash) !== -1) {
    showTab(initialHash);
  } else {
    showTab('dashboard');
  }

  connect();
})();
