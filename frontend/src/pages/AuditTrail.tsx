import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { listAudit } from "@/services/audit";
import { Card } from "@/components/ui/Card";
import { PageHeader } from "@/components/ui/PageHeader";
import { Spinner } from "@/components/ui/Spinner";
import { Badge } from "@/components/ui/Badge";
import { Modal } from "@/components/ui/Modal";
import { formatDate, truncate } from "@/utils/format";

export default function AuditTrail() {
  const [entity, setEntity] = useState("");
  const [action, setAction] = useState("");
  const [selected, setSelected] = useState<any>(null);

  const list = useQuery({
    queryKey: ["audit", entity, action],
    queryFn: () => listAudit({ entity_type: entity || undefined, action: action || undefined, limit: 200 }),
  });

  return (
    <>
      <PageHeader title="Audit Trail" subtitle="Every significant action with before / after state." />

      <Card>
        <div className="toolbar">
          <label className="form-row form-row-inline">
            <span>Entity</span>
            <select className="input" value={entity} onChange={(e) => setEntity(e.target.value)}>
              <option value="">All</option>
              <option value="recommendation_run">Recommendation run</option>
              <option value="agent_message">Agent message</option>
              <option value="model_registry">Model registry</option>
              <option value="purchase_order">Purchase order</option>
            </select>
          </label>
          <label className="form-row form-row-inline">
            <span>Action contains</span>
            <input className="input" value={action} onChange={(e) => setAction(e.target.value)} />
          </label>
        </div>

        {list.isLoading ? <Spinner /> : (
          <table className="table">
            <thead>
              <tr>
                <th>When</th>
                <th>Actor</th>
                <th>Action</th>
                <th>Entity</th>
                <th>Trace</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {(list.data ?? []).length === 0 && (
                <tr><td colSpan={6} className="table-empty">No audit events.</td></tr>
              )}
              {(list.data ?? []).map((a) => (
                <tr key={a.id}>
                  <td>{formatDate(a.created_at)}</td>
                  <td>{a.actor_name ?? (a.actor_id ? `#${a.actor_id}` : "—")}</td>
                  <td><Badge tone="info">{a.action}</Badge></td>
                  <td>{a.entity_type}{a.entity_id ? ` #${a.entity_id}` : ""}</td>
                  <td><code>{a.trace_id ? truncate(a.trace_id, 12) : "—"}</code></td>
                  <td><button className="btn btn-ghost btn-sm" onClick={() => setSelected(a)}>View</button></td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Card>

      <Modal open={!!selected} onClose={() => setSelected(null)} title={`Audit #${selected?.id ?? ""}`}>
        {selected && (
          <div>
            <div className="json-panel-title">Before</div>
            <pre className="json-view">{JSON.stringify(selected.before, null, 2)}</pre>
            <div className="json-panel-title">After</div>
            <pre className="json-view">{JSON.stringify(selected.after, null, 2)}</pre>
          </div>
        )}
      </Modal>
    </>
  );
}