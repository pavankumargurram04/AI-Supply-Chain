import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { listAlerts, getUsage } from "@/services/monitoring";
import { Card } from "@/components/ui/Card";
import { PageHeader } from "@/components/ui/PageHeader";
import { Spinner } from "@/components/ui/Spinner";
import { Badge } from "@/components/ui/Badge";
import { Kpi } from "@/components/ui/Kpi";
import { Button } from "@/components/ui/Button";
import { formatDate } from "@/utils/format";

export default function Monitoring() {
  const [status, setStatus] = useState("open");
  const [severity, setSeverity] = useState("");
  const [days, setDays] = useState(7);

  const alerts = useQuery({
    queryKey: ["alerts", status, severity],
    queryFn: () => listAlerts(status || undefined, severity || undefined),
  });

  const usage = useQuery({
    queryKey: ["usage", days],
    queryFn: () => getUsage(days),
  });

  return (
    <>
      <PageHeader title="Monitoring" subtitle="Alerts, drift signals, and operational usage." />

      {usage.data && (
        <div className="grid grid-cols-4">
          <Kpi label={`Recommendations (${days}d)`} value={usage.data.recommendations.total} />
          <Kpi label={`Predictions (${days}d)`} value={usage.data.predictions.total} />
          <Kpi label={`Audit events (${days}d)`} value={usage.data.audit_events.total} />
          <Kpi label="Order created" value={usage.data.approval_funnel.order_created} />
        </div>
      )}

      <Card title="Alerts" className="mt-4"
        actions={<Button variant="secondary" onClick={() => alerts.refetch()}>Refresh</Button>}>
        <div className="toolbar">
          <label className="form-row form-row-inline">
            <span>Status</span>
            <select className="input" value={status} onChange={(e) => setStatus(e.target.value)}>
              <option value="">All</option>
              <option value="open">Open</option>
              <option value="acknowledged">Acknowledged</option>
              <option value="resolved">Resolved</option>
              <option value="dismissed">Dismissed</option>
            </select>
          </label>
          <label className="form-row form-row-inline">
            <span>Severity</span>
            <select className="input" value={severity} onChange={(e) => setSeverity(e.target.value)}>
              <option value="">All</option>
              <option value="info">Info</option>
              <option value="warning">Warning</option>
              <option value="critical">Critical</option>
            </select>
          </label>
        </div>

        {alerts.isLoading ? <Spinner /> : (
          <table className="table">
            <thead>
              <tr>
                <th>Severity</th>
                <th>Source</th>
                <th>Title</th>
                <th>Message</th>
                <th>Status</th>
                <th>Created</th>
              </tr>
            </thead>
            <tbody>
              {(alerts.data ?? []).length === 0 && (
                <tr><td colSpan={6} className="table-empty">No alerts match the filter.</td></tr>
              )}
              {(alerts.data ?? []).map((a) => (
                <tr key={a.id}>
                  <td><Badge tone={a.severity === "critical" ? "danger" : a.severity === "warning" ? "warning" : "info"}>{a.severity}</Badge></td>
                  <td><code>{a.source}</code></td>
                  <td>{a.title}</td>
                  <td>{a.message}</td>
                  <td><Badge>{a.status}</Badge></td>
                  <td>{formatDate(a.created_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Card>
    </>
  );
}