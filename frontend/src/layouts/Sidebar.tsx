import { NavLink } from "react-router-dom";
import { useAuthStore } from "@/state/authStore";
import { hasRole } from "@/utils/roles";
import clsx from "clsx";

interface NavItem {
  to: string;
  label: string;
  roles?: string[];
}

const NAV: NavItem[] = [
  { to: "/", label: "Dashboard" },
  { to: "/products", label: "Products" },
  { to: "/demand", label: "Demand Forecast" },
  { to: "/supplier-risk", label: "Supplier Risk" },
  { to: "/inventory", label: "Inventory" },
  { to: "/procurement", label: "Procurement" },
  { to: "/orders", label: "Orders" },
  { to: "/shipments", label: "Shipments" },
  { to: "/warehouses", label: "Warehouses" },
  { to: "/scenarios", label: "Scenario Lab" },
  { to: "/agents", label: "Multi-Agent" },
  { to: "/models", label: "Model Center" },
  { to: "/training", label: "Training Center", roles: ["admin"] },
  { to: "/monitoring", label: "Monitoring" },
  { to: "/reports", label: "Reports" },
  { to: "/audit", label: "Audit Trail" },
];

export function Sidebar() {
  const user = useAuthStore((s) => s.user);
  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <div className="brand-mark">AI</div>
        <div>
          <div className="brand-title">Supply Chain</div>
          <div className="brand-sub">Enterprise</div>
        </div>
      </div>
      <nav className="sidebar-nav">
        {NAV.filter((n) => !n.roles || hasRole(user?.role, ...(n.roles as any))).map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === "/"}
            className={({ isActive }) => clsx("nav-item", isActive && "nav-item-active")}
          >
            {item.label}
          </NavLink>
        ))}
      </nav>
    </aside>
  );
}