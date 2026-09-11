import type { Report } from "./types";

export async function fetchReport(filename: string): Promise<Report> {
  const response = await fetch(`/runs/${encodeURIComponent(filename)}`);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return await response.json() as Report;
}
