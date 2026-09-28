import { useQuery } from "@tanstack/react-query";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { listRecommendations } from "@/services/procurement";
import { listAlerts, getUsage } from "@/services/monitoring";
import { listModels } from "@/services/models";
import { listSuppliers } from "@/services/suppliers";
import { Card } from "@/components/ui/Card";
import { Kpi } from "@/components/ui/Kpi";
import { PageHeader } from "@/components/ui/PageHeader";
import { Spinner } from "@/components/ui/Spinner";
import { Badge } from "@/components/ui/Badge";
import { formatCurrency, formatNumber, formatPercent, formatDateShort } from "@/utils/format";

const SEVERITY_COLORS: Record<string, string> = {
  info: "#0284c7",
  warning: "#d97706",
  critical: "#dc2626",
};

export default function Dashboard() {
  const recs = useQuery({
    queryKey: ["dashboard", "recommendations"],
    queryFn: () => listRecommendations(),
  });

  const alerts = useQuery({
    queryKey: ["dashboard", "alerts"],
    queryFn: () => listAlerts("open"),
  });

  const usage = useQuery({
    queryKey: ["dashboard", "usage"],
    queryFn: () => getUsage(7),
  });

  const models = useQuery({
    queryKey: ["dashboard", "models"],
    queryFn: () => listModels(),
  });

  const suppliers = useQuery({
    queryKey: ["dashboard", "suppliers"],
    queryFn: () => listSuppliers(),
  });

  const loading =
    recs.isLoading || alerts.isLoading || usage.isLoading || models.isLoading || suppliers.isLoading;

  if (loading) return <Spinner label="Loading dashboard…" />;

  const recsList = recs.data ?? [];
  const alertsList = alerts.data ?? [];
  const usageSnap = usage.data;
  const modelsList = models.data ?? [];
  const suppliersList = suppliers.data ?? [];

  const pendingApprovals = recsList.filter((r) => r.status === "pending_review").length;
  const openPOCost = recsList
    .filter((r) => r.status === "approved" || r.status === "order_created")
    .reduce((s, r) => s + (r.expected_cost ?? 0), 0);

  const avgService =
    recsList.length > 0
      ? recsList.reduce((s, r) => s + (r.expected_service_level ?? 0), 0) / recsList.length
      : 0;

  const avgStockout =
    recsList.length > 0
      ? recsList.reduce((s, r) => s + (r.stockout_probability ?? 0), 0) / recsList.length
      : 0;

  // Charts -----------------------------------------------------------------
  const alertsBySeverity = ["info", "warning", "critical"].map((sev) => ({
    severity: sev,
    count: alertsList.filter((a) => a.severity === sev).length,
  }));

  const statusCounts = Object.entries(usageSnap?.recommendations.by_status ?? {}).map(
    ([status, count]) => ({ status, count }),
  );

  const supplierReliability = ["premium", "reliable", "standard", "poor"].map((r) => ({
    class: r,
    count: suppliersList.filter((s) => s.reliability_class === r).length,
  }));

  const modelStatusCounts = ["trained", "validated", "staging", "production", "archived", "rejected"].map(
    (s) => ({ status: s, count: modelsList.filter((m) => m.status === s).length }),
  );

  const recentRecs = recsList.slice(0, 6);

  return (
    <>
      <PageHeader title="Dashboard" subtitle="Operational overview — values are derived from live API data" />

      <div className="grid grid-cols-4">
        <Kpi label="Pending approvals" value={pendingApprovals} tone={pendingApprovals > 0 ? "warn" : "default"} />
        <Kpi label="Approved order value" value={formatCurrency(openPOCost)} />
        <Kpi label="Avg. expected service" value={formatPercent(avgService)} tone={avgService > 0.9 ? "good" : "warn"} />
        <Kpi label="Avg. stockout probability" value={formatPercent(avgStockout)} tone={avgStockout > 0.3 ? "bad" : "good"} />
      </div>

      <div className="grid grid-cols-4" style={{ marginTop: 16 }}>
        <Kpi label="Open alerts" value={alertsList.length} tone={alertsList.length > 0 ? "warn" : "good"} />
        <Kpi label="Recommendations (7d)" value={usageSnap?.recommendations.total ?? 0} />
        <Kpi label="Predictions (7d)" value={usageSnap?.predictions.total ?? 0} />
        <Kpi label="Audit events (7d)" value={usageSnap?.audit_events.total ?? 0} />
      </div>

      <div className="grid grid-cols-2" style={{ marginTop: 16 }}>
        <Card title="Alerts by severity">
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={alertsBySeverity}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="severity" />
              <YAxis allowDecimals={false} />
              <Tooltip />
              <Bar dataKey="count">
                {alertsBySeverity.map((a) => (
                  <Cell key={a.severity} fill={SEVERITY_COLORS[a.severity] ?? "#8884d8"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <Card title="Recommendation status (7 days)">
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={statusCounts}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="status" />
              <YAxis allowDecimals={false} />
              <Tooltip />
              <Bar dataKey="count" fill="#2563eb" />
            </BarChart>
          </ResponsiveContainer>
        </Card>
      </div>

      <div className="grid grid-cols-2" style={{ marginTop: 16 }}>
        <Card title="Supplier reliability distribution">
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={supplierReliability}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="class" />
              <YAxis allowDecimals={false} />
              <Tooltip />
              <Bar dataKey="count" fill="#059669" />
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <Card title="Model registry status">
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={modelStatusCounts}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="status" />
              <YAxis allowDecimals={false} />
              <Tooltip />
              <Bar dataKey="count" fill="#7c3aed" />
            </BarChart>
          </ResponsiveContainer>
        </Card>
      </div>

      <Card title="Recent recommendations" className="mt-4">
        <table className="table">
          <thead>
            <tr>
              <th>Trace</th>
              <th>Status</th>
              <th>Qty</th>
              <th>Cost</th>
              <th>Service</th>
              <th>Stockout</th>
              <th>Created</th>
            </tr>
          </thead>
          <tbody>
            {recentRecs.length === 0 && (
              <tr>
                <td colSpan={7} className="table-empty">No recommendations yet.</td>
              </tr>
            )}
            {recentRecs.map((r) => (
              <tr key={r.id}>
                <td><code>{r.trace_id.slice(0, 10)}…</code></td>
                <td><Badge tone={r.status === "approved" ? "success" : r.status === "rejected" ? "danger" : "info"}>{r.status}</Badge></td>
                <td>{formatNumber(r.final_quantity)}</td>
                <td>{formatCurrency(r.expected_cost)}</td>
                <td>{formatPercent(r.expected_service_level)}</td>
                <td>{formatPercent(r.stockout_probability)}</td>
                <td>{formatDateShort(r.created_at)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>
    </>
  );
}