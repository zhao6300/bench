export interface Run {
  filename: string;
  size_bytes: number;
  modified_at: string | null;
}

export interface Case {
  id?: string;
  name?: string;
  status?: string;
  duration_seconds?: number;
  matrix?: {
    requests?: { concurrency?: number } | number | null;
  };
  result?: {
    metrics?: Record<string, number | string | null>;
  };
}

export interface Report {
  cases?: Case[];
  summary?: Record<string, number | string | null>;
  environment?: {
    host_inventory?: {
      cpu?: { model?: string; logical_cores?: number; physical_cores?: number };
      memory?: { total_bytes?: number };
      accelerators?: { devices?: { name?: string }[] };
    };
  };
  suite?: {
    name?: string;
    started_at?: string;
    duration_seconds?: number;
  };
}
