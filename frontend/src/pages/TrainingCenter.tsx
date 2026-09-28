import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { enqueueTraining, listJobs, getJob } from "@/services/training";
import { Card } from "@/components/ui/Card";
import { PageHeader } from "@/components/ui/PageHeader";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import { Spinner } from "@/components/ui/Spinner";
import { Modal } from "@/components/ui/Modal";
import { extractErrorMessage } from "@/services/api";
import { formatDate } from "@/utils/format";

export default function TrainingCenter() {
  const qc = useQueryClient();
  const [family, setFamily] = useState("demand_lstm");
  const [mode, setMode] = useState("dev");
  const [seed, setSeed] = useState(42);
  const [status, setStatus] = useState("");
  const [selected, setSelected] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  const jobs = useQuery({
    queryKey: ["training-jobs", status],
    queryFn: () => listJobs(status || undefined),
    refetchInterval: 5000,
  });

  const detail = useQuery({
    queryKey: ["training-job", selected],
    queryFn: () => getJob(selected!),
    enabled: selected !== null,
    refetchInterval: 3000,
  });

  const mutation = useMutation({
    mutationFn: () => enqueueTraining(family, mode, seed),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["training-jobs"] }),
    onError: (e) => setError(extractErrorMessage(e)),
  });

  return (
    <>
      <PageHeader title="Training Center" subtitle="Queue new training jobs and monitor their progress (admin only)." />

      <Card title="Enqueue training">
        <div className="toolbar">
          <label className="form-row form-row-inline">
            <span>Family</span>
            <select className="input" value={family} onChange={(e) => setFamily(e.target.value)}>
              <option value="demand_lstm">Demand LSTM</option>
              <option value="supplier_xgb">Supplier XGBoost</option>
              <option value="inventory_dqn">Inventory DQN</option>
            </select>
          </label>
          <label className="form-row form-row-inline">
            <span>Mode</span>
            <select className="input" value={mode} onChange={(e) => setMode(e.target.value)}>
              <option value="dev">dev</option>
              <option value="research">research</option>
              <option value="stress">stress</option>
            </select>
          </label>
          <label className="form-row form-row-inline">
            <span>Seed</span>
            <input className="input" type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value))} />
          </label>
          <Button onClick={() => { setError(null); mutation.mutate(); }} loading={mutation.isPending}>
            Enqueue job
          </Button>
        </div>
        {error && <div className="form-error">{error}</div>}
        {mutation.data && (
          <div className="muted-note">
            Job #{mutation.data.job_id} — {mutation.data.dispatched ? "dispatched to worker" : "queued (worker may be offline)"}
          </div>
        )}
      </Card>

      <Card title="Jobs" className="mt-4">
        <div className="toolbar">
          <label className="form-row form-row-inline">
            <span>Status</span>
            <select className="input" value={status} onChange={(e) => setStatus(e.target.value)}>
              <option value="">All</option>
              <option value="queued">Queued</option>
              <option value="running">Running</option>
              <option value="succeeded">Succeeded</option>
              <option value="failed">Failed</option>
            </select>
          </label>
          <Button variant="secondary" onClick={() => jobs.refetch()}>Refresh</Button>
        </div>

        {jobs.isLoading ? <Spinner /> : (
          <table className="table">
            <thead>
              <tr>
                <th>ID</th>
                <th>Family</th>
                <th>Status</th>
                <th>Seed</th>
                <th>Created</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {(jobs.data ?? []).length === 0 && (
                <tr><td colSpan={6} className="table-empty">No training jobs.</td></tr>
              )}
              {(jobs.data ?? []).map((j: any) => (
                <tr key={j.id}>
                  <td>#{j.id}</td>
                  <td>{j.model_family}</td>
                  <td><Badge tone={j.status === "succeeded" ? "success" : j.status === "failed" ? "danger" : "info"}>{j.status}</Badge></td>
                  <td>{j.seed}</td>
                  <td>{formatDate(j.created_at)}</td>
                  <td><Button size="sm" variant="ghost" onClick={() => setSelected(j.id)}>View</Button></td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Card>

      <Modal open={selected !== null} onClose={() => setSelected(null)} title={`Training job #${selected ?? ""}`}>
        {detail.isLoading ? <Spinner /> : detail.data ? (
          <div>
            <div className="kv-grid">
              <div><span>Family</span><b>{detail.data.model_family}</b></div>
              <div><span>Status</span><b>{detail.data.status}</b></div>
              <div><span>Seed</span><b>{detail.data.seed}</b></div>
              <div><span>Started</span><b>{formatDate(detail.data.started_at)}</b></div>
              <div><span>Finished</span><b>{formatDate(detail.data.finished_at)}</b></div>
            </div>
            {detail.data.error_message && <div className="form-error">{detail.data.error_message}</div>}
            <div className="json-panel-title">Config</div>
            <pre className="json-view">{JSON.stringify(detail.data.config, null, 2)}</pre>
            <div className="json-panel-title">Metrics</div>
            <pre className="json-view">{JSON.stringify(detail.data.metrics, null, 2)}</pre>
          </div>
        ) : null}
      </Modal>
    </>
  );
}