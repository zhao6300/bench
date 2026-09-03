/* Zero-dependency unit tests for the dashboard server (node:test). */
'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');

const {
  makeRun,
  applyIngest,
  markIdle,
  snapshot,
  isReportName,
  findReportPath,
  listExamples,
  listReports,
} = require('./server.js');

function freshRun(config) {
  markIdle();
  return makeRun(config);
}

function tokenOf(run) {
  return run._token;
}

/* ── run id generation ─────────────────────────────────────────────────────── */

test('makeRun produces unique ids and stores initial state', () => {
  const runA = freshRun({name: 'a', cases: [{}, {}]});
  const runB = freshRun({name: 'b', cases: [{}]});
  assert.notStrictEqual(runA.id, runB.id);
  assert.match(runA.id, /^r\d{12}-\d+$/);
  assert.strictEqual(runA.case_total, 2);
  assert.strictEqual(runA.status, 'queued');
  assert.strictEqual(runB.case_total, 1);
});

test('makeRun without a cases array still creates a valid record', () => {
  const run = freshRun({});
  assert.strictEqual(run.case_total, 0);
  assert.strictEqual(run.status, 'queued');
});

/* ── applyIngest state transitions ─────────────────────────────────────────── */

test('case_started updates current_case, position and marks run running', () => {
  const run = freshRun({});
  const data = {event: 'case_started', case_name: 'a', scenario: 'single', position: 2, total_cases: 5};
  applyIngest(data, tokenOf(run));
  assert.strictEqual(run.status, 'running');
  assert.deepStrictEqual(run.current_case, {name: 'a', scenario: 'single'});
  assert.strictEqual(run.case_position, 2);
  assert.strictEqual(run.case_total, 5);
  assert.strictEqual(run.stage, null);
});

test('round_started/resets and request_finished accumulates counters', () => {
  const run = freshRun({});
  applyIngest({event: 'round_started', round_total: 4, concurrency: 2}, tokenOf(run));
  assert.deepStrictEqual(run.round, {total_requests: 4, concurrency: 2, completed: 0, succeeded: 0, failed: 0});
  applyIngest({event: 'request_finished', completed: 1, succeeded: 1, failed: 0}, tokenOf(run));
  applyIngest({event: 'request_finished', completed: 2, succeeded: 1, failed: 1}, tokenOf(run));
  assert.strictEqual(run.round.completed, 2);
  assert.strictEqual(run.round.succeeded, 1);
  assert.strictEqual(run.round.failed, 1);
  assert.strictEqual(run.round.total_requests, 4);
});

test('stage transitions originate from stage_started / stage_finished', () => {
  const run = freshRun({});
  applyIngest({event: 'stage_started', stage: 'warmup'}, tokenOf(run));
  assert.strictEqual(run.stage, 'warmup');
  applyIngest({event: 'stage_finished', stage: 'warmup'}, tokenOf(run));
  assert.strictEqual(run.stage, null);
});

test('final_results stores report and refreshes suite name', () => {
  const run = freshRun({name: 'original'});
  applyIngest({event: 'final_results', report: {suite: {name: 'new-suite-name'}}}, tokenOf(run));
  assert.deepStrictEqual(run.final_report, {suite: {name: 'new-suite-name'}});
  assert.strictEqual(run.name, 'new-suite-name');
});

test('events beyond MAX_RUN_EVENTS are ring-buffered', () => {
  const run = freshRun({});
  for (let i = 0; i < 500; i += 1) {
    applyIngest({event: 'event', message: `m${i}`}, tokenOf(run));
  }
  assert.ok(run.events.length <= 400);
});

test('ingest with unknown token routes to null and does not touch any run', () => {
  const run = freshRun({});
  const before = run.events.length;
  applyIngest({event: 'event', message: 'orphan'}, 'not-a-real-token');
  assert.strictEqual(run.events.length, before);
});

/* ── report name / path resolution ─────────────────────────────────────────── */

test('isReportName accepts only recognised report JSON names', () => {
  assert.strictEqual(isReportName('benchmark-report.json'), true);
  assert.strictEqual(isReportName('benchmark-report-2026-01-01.json'), true);
  assert.strictEqual(isReportName('smoke-report-abc.json'), true);
  assert.strictEqual(isReportName('README.json'), false);
  assert.strictEqual(isReportName('smoke.data'), false);
  assert.strictEqual(isReportName('benchmark-report.txt'), false);
});

test('isReportName rejects path traversal shapes', () => {
  assert.strictEqual(isReportName('..'), false);
  assert.strictEqual(isReportName('foo/../bar.json'), false);
  assert.strictEqual(isReportName('a/b/c.json'), false);
  assert.strictEqual(isReportName('..\\bar.json'), false);
});

test('findReportPath returns null for non-existent files', () => {
  assert.strictEqual(findReportPath(' exceedingly-unusual-name.json '), null);
});

/* ── example listing ───────────────────────────────────────────────────────── */

test('listExamples returns at least one parseable example when run from repo root', () => {
  const examples = listExamples();
  assert.ok(examples.length >= 1, 'expected at least one example');
  const first = examples[0];
  assert.ok(first.name);
  assert.strictEqual(typeof first.config, 'object');
});

/* ── summary shape ─────────────────────────────────────────────────────────── */

test('snapshot exposes runs list, events ring buffer and is JSON-safe', () => {
  markIdle();
  const run = makeRun({name: 's', cases: []});
  applyIngest({event: 'case_started', case_name: 'x', scenario: 'single', position: 1, total_cases: 1}, tokenOf(run));
  const snap = snapshot();
  assert.ok(Array.isArray(snap.runs));
  assert.ok(Array.isArray(snap.events));
  assert.strictEqual(snap.active_run_id, run.id);
  assert.strictEqual(snap.total_runs, 1);
  JSON.stringify(snap);
});
