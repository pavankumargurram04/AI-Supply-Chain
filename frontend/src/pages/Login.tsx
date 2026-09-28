import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { login, me } from "@/services/auth";
import { useAuthStore } from "@/state/authStore";
import { extractErrorMessage } from "@/services/api";
import { Button } from "@/components/ui/Button";

export default function Login() {
  const [username, setUsername] = useState("admin");
  const [password, setPassword] = useState("admin");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const navigate = useNavigate();
  const setTokens = useAuthStore((s) => s.setTokens);
  const setUser = useAuthStore((s) => s.setUser);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setBusy(true);
    try {
      const t = await login(username, password);
      setTokens(t.access_token, t.refresh_token);
      const u = await me();
      setUser(u);
      navigate("/");
    } catch (err) {
      setError(extractErrorMessage(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="login-page">
      <form className="login-card" onSubmit={submit}>
        <div className="login-brand">
          <div className="brand-mark">AI</div>
          <div>
            <div className="login-title">AI Supply Chain Enterprise</div>
            <div className="login-sub">Decision Intelligence Platform</div>
          </div>
        </div>

        <label className="form-row">
          <span>Username</span>
          <input
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            autoComplete="username"
            required
          />
        </label>

        <label className="form-row">
          <span>Password</span>
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            autoComplete="current-password"
            required
          />
        </label>

        {error && <div className="form-error">{error}</div>}

        <Button type="submit" loading={busy} size="lg">
          Sign in
        </Button>

        <div className="login-hint">
          Default credentials created by <code>scripts/bootstrap_database.py</code>: <b>admin / admin</b>.
        </div>
      </form>
    </div>
  );
}