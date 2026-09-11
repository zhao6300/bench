import type { Report } from "./types";

function buildReportUrl(filename: string): string {
  if (!filename.endsWith(".json")) throw new Error("无效报告文件");
  return `/runs/${encodeURIComponent(filename)}`;
}

export async function fetchReport(filename: string, signal?: AbortSignal): Promise<Report> {
  const response = await fetch(buildReportUrl(filename), { cache: "no-store", signal });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return await response.json() as Report;
}
