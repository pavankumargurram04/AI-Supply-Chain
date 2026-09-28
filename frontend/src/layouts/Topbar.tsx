import { useNavigate } from "react-router-dom";
import { useAuthStore } from "@/state/authStore";
import { Button } from "@/components/ui/Button";

export function Topbar() {
  const user = useAuthStore((s) => s.user);
  const clear = useAuthStore((s) => s.clear);
  const navigate = useNavigate();

  const logout = () => {
    clear();
    navigate("/login");
  };

  return (
    <header className="topbar">
      <div />
      <div className="topbar-user">
        {user && (
          <>
            <div className="user-meta">
              <div className="user-name">{user.full_name || user.username}</div>
              <div className="user-role">{user.role}</div>
            </div>
            <Button variant="ghost" size="sm" onClick={logout}>Log out</Button>
          </>
        )}
      </div>
    </header>
  );
}