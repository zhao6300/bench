import type { Report } from "./types";

export interface AuthStatus {
  authenticated: boolean;
  username?: string;
}

function buildReportUrl(filename: string): string {
  if (!filename.endsWith(".json")) throw new Error("无效报告文件");
  return `/runs/${encodeURIComponent(filename)}`;
}

async function responseMessage(response: Response, fallback: string): Promise<string> {
  try {
    const payload = await response.json() as { message?: string };
    return typeof payload.message === "string" ? payload.message : fallback;
  } catch {
    return fallback;
  }
}

export async function fetchAuthStatus(): Promise<AuthStatus> {
  const response = await fetch("/api/auth/session", { cache: "no-store" });
  const payload = await response.json() as AuthStatus;
  if (!response.ok) {
    if (response.status === 401) return { authenticated: false };
    throw new Error(`HTTP ${response.status}`);
  }
  return payload;
}

export async function login(username: string, password: string): Promise<AuthStatus> {
  const response = await fetch("/api/auth/session", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
    cache: "no-store",
  });
  if (!response.ok) throw new Error(await responseMessage(response, "账号或密码错误"));
  return await response.json() as AuthStatus;
}

export async function logout(): Promise<void> {
  await fetch("/api/auth/session", { method: "DELETE", cache: "no-store" });
  window.location.assign("/");
}

export async function fetchReport(filename: string, signal?: AbortSignal): Promise<Report> {
  const response = await fetch(buildReportUrl(filename), { cache: "no-store", signal });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return await response.json() as Report;
}
