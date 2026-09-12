import { useState } from "react";
import { login } from "../services";

interface AuthPanelProps {
  onAuthenticated: (username: string) => void;
}

export default function AuthPanel({ onAuthenticated }: AuthPanelProps) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (submitting) return;
    setSubmitting(true);
    setError("");
    try {
      const status = await login(username, password);
      onAuthenticated(status.username ?? username);
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "登录失败");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="auth-shell">
      <form className="auth-panel" onSubmit={handleSubmit}>
        <div className="brand-icon auth-brand">LB</div>
        <h1>LLM Benchmark</h1>
        <p>请登录后查看报告</p>
        <label htmlFor="auth-username">用户名</label>
        <input
          id="auth-username"
          type="text"
          autoComplete="username"
          value={username}
          onChange={(event) => setUsername(event.target.value)}
          required
        />
        <label htmlFor="auth-password">密码</label>
        <input
          id="auth-password"
          type="password"
          autoComplete="current-password"
          value={password}
          onChange={(event) => setPassword(event.target.value)}
          required
        />
        <button type="submit" disabled={submitting}>
          {submitting ? "登录中…" : "登录"}
        </button>
        {error ? <p className="auth-error">{error}</p> : null}
      </form>
    </div>
  );
}
