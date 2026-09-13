export type MetricValue = number | string | boolean | null;

export interface Run {
  filename: string;
  size_bytes: number;
  modified_at: string | null;
}

export interface Case {
  id?: string;
  name?: string;
  status?: string;
  case_key?: string;
  duration_seconds?: number;
  params?: Record<string, MetricValue | string[]>;
  matrix?: {
    requests?: { concurrency?: number; count?: number } | number | null;
  };
  result?: {
    metrics?: Record<string, unknown>;
  };
}

export interface SuiteMetadata {
  [key: string]: unknown;
}

export interface ReportSuite {
  name?: string;
  description?: string;
  config_file?: string;
  started_at?: string;
  finished_at?: string;
  duration_seconds?: number;
  run_state?: string;
  metadata?: SuiteMetadata;
  failure_policy?: string;
  execution_plan_sha256?: string;
}

export interface EnvironmentHost {
  cpu?: { model?: string; logical_cores?: number; physical_cores?: number };
  memory?: { total_bytes?: number };
  operating_system?: { system?: string; release?: string; machine?: string };
  accelerators?: { devices?: { name?: string }[] };
}

export interface ReportEnvironment {
  python?: string;
  platform?: string;
  hostname?: string;
  host_inventory?: EnvironmentHost;
}

export interface Report {
  schema_version?: number;
  cases?: Case[];
  summary?: Record<string, MetricValue>;
  environment?: ReportEnvironment;
  suite?: ReportSuite;
}

export interface ReportPair {
  reportA: Report | null;
  reportB: Report | null;
}
