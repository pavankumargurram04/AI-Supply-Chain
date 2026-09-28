import { useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { listModels, promote, rollback, getModel } from "@/services/models";
import { Card } from "@/components/ui/Card";
import { PageHeader } from "@/components/ui/PageHeader";
import { Spinner } from "@/components/ui/Spinner";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Modal } from "@/components/ui/Modal";
import { extractErrorMessage } from "@/services/api";
import { formatDate } from "@/utils/format";

export default function ModelCenter() {
  const qc = useQueryClient();
  const [family, setFamily] = useState("");
  const [status, setStatus] = useState("");
  const [inspect, setInspect] = useState<{ name: string; version: string } | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const list = useQuery({
    queryKey: ["models", family, status],
    queryFn: () => listModels(family || undefined, status || undefined),
  });

  const detail = useQuery({
    queryKey: ["model", inspect?.name, inspect?.version],
    queryFn: () => getModel(inspect!.name, inspect!.version),
    enabled: !!inspect,
  });

  async function onPromote(name: string, version: string) {
    setBusy(`${name}/${version}`);
    setError(null);
    try {
      await promote(name, version);
      await qc.invalidateQueries({ queryKey: ["models"] });
    } catch (e) { setError(extractErrorMessage(e)); }
    finally { setBusy(null); }
  }

  async function onRollback(name: string) {
    setBusy(name);
    setError(null);
    try {
      await rollback(name, "manual rollback from Model Center");
      await qc.invalidateQueries({ queryKey: ["models"] });
    } catch (e) { setError(extractErrorMessage(e)); }
    finally { setBusy(null); }
  }

  return (
    <>
      <PageHeader title="Model Center" subtitle="Registry of every trained model version with lifecycle controls." />

      <Card>
        <div className="toolbar">
          <label className="form-row form-row-inline">
            <span>Family</span>
            <select className="input" value={family} onChange={(e) => setFamily(e.target.value)}>
              <option value="">All</option>
              <option value="demand_lstm">Demand LSTM</option>
              <option value="supplier_xgb">Supplier XGBoost</option>
              <option value="inventory_dqn">Inventory DQN</option>
            </select>
          </label>
          <label className="form-row form-row-inline">
            <span>Status</span>
            <select className="input" value={status} onChange={(e) => setStatus(e.target.value)}>
              <option value="">All</option>
              <option value="trained">Trained</option>
              <option value="validated">Validated</option>
              <option value="staging">Staging</option>
              <option value="production">Production</option>
              <option value="archived">Archived</option>
              <option value="rejected">Rejected</option>
            </select>
          </label>
        </div>

        {error && <div className="form-error">{error}</div>}

        {list.isLoading ? <Spinner /> : (
          <table className="table">
            <thead>
              <tr>
                <th>Model</th>
                <th>Family</th>
                <th>Version</th>
                <th>Status</th>
                <th>Feature version</th>
                <th>Trained</th>
                <th style={{ width: 260 }}>Actions</th>
              </tr>
            </thead>
            <tbody>
              {(list.data ?? []).length === 0 && (
                <tr><td colSpan={7} className="table-empty">No models registered yet.</td></tr>
              )}
              {(list.data ?? []).map((m) => (
                <tr key={m.id}>
                  <td><code>{m.model_name}</code></td>
                  <td>{m.family}</td>
                  <td><code>{m.version}</code></td>
                  <td><Badge tone={m.status === "production" ? "success" : m.status === "archived" ? "default" : "info"}>{m.status}</Badge></td>
                  <td><code>{m.feature_version ?? "—"}</code></td>
                  <td>{formatDate(m.trained_at)}</td>
                  <td>
                    <Button size="sm" variant="ghost" onClick={() => setInspect({ name: m.model_name, version: m.version })}>View</Button>{" "}
                    {m.status !== "production" && (
                      <Button size="sm" onClick={() => onPromote(m.model_name, m.version)} loading={busy === `${m.model_name}/${m.version}`}>Promote</Button>
                    )}{" "}
                    {m.status === "production" && (
                      <Button size="sm" variant="danger" onClick={() => onRollback(m.model_name)} loading={busy === m.model_name}>Rollback</Button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Card>

      <Modal open={!!inspect} onClose={() => setInspect(null)} title={inspect ? `${inspect.name} · ${inspect.version}` : ""}>
        {detail.isLoading ? <Spinner /> : detail.data ? (
          <div>
            <div className="json-panel-title">Metrics</div>
            <pre className="json-view">{JSON.stringify(detail.data.metrics, null, 2)}</pre>
            <div className="json-panel-title">Hyperparameters</div>
            <pre className="json-view">{JSON.stringify(detail.data.hyperparameters, null, 2)}</pre>
            <div className="json-panel-title">Artifact</div>
            <pre className="json-view">{detail.data.artifact_uri ?? "—"}</pre>
          </div>
        ) : null}
      </Modal>
    </>
  );
}