import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { useQuery } from "@tanstack/react-query";
import { listProducts } from "@/services/products";
import { listWarehouses } from "@/services/warehouses";
import { runAgents, type AgentRun } from "@/services/agents";
import { Card } from "@/components/ui/Card";
import { PageHeader } from "@/components/ui/PageHeader";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import { extractErrorMessage } from "@/services/api";
import { formatDate } from "@/utils/format";

const AGENT_ORDER = ["demand", "supplier", "inventory", "procurement"];

const AGENT_LABEL: Record<string, string> = {
  demand: "Demand Agent",
  supplier: "Supplier Agent",
  inventory: "Inventory Agent",
  procurement: "Procurement Agent",
};

export default function MultiAgent() {
  const products = useQuery({ queryKey: ["products", "all"], queryFn: () => listProducts(undefined, 500) });
  const warehouses = useQuery({ queryKey: ["warehouses"], queryFn: () => listWarehouses(100) });

  const [sku, setSku] = useState("");
  const [wh, setWh] = useState("");
  const [run, setRun] = useState<AgentRun | null>(null);
  const [open, setOpen] = useState<string | null>(null);

  const mutation = useMutation({
    mutationFn: () => runAgents({ product_sku: sku, warehouse_code: wh }),
    onSuccess: (r) => setRun(r),
  });

  const messagesByAgent = new Map<string, any>();
  if (run) for (const m of run.messages) messagesByAgent.set(m.agent, m);

  return (
    <>
      <PageHeader title="Multi-Agent" subtitle="Orchestrated decision pipeline with structured evidence at every step." />

      <Card>
        <div className="toolbar">
          <label className="form-row form-row-inline">
            <span>Product</span>
            <select className="input" value={sku} onChange={(e) => setSku(e.target.value)}>
              <option value="">Select…</option>
              {products.data?.map((p) => <option key={p.id} value={p.sku}>{p.sku}</option>)}
            </select>
          </label>
          <label className="form-row form-row-inline">
            <span>Warehouse</span>
            <select className="input" value={wh} onChange={(e) => setWh(e.target.value)}>
              <option value="">Select…</option>
              {warehouses.data?.map((w) => <option key={w.id} value={w.code}>{w.code}</option>)}
            </select>
          </label>
          <Button onClick={() => mutation.mutate()} loading={mutation.isPending} disabled={!sku || !wh}>
            Run agents
          </Button>
        </div>
        {mutation.error && <div className="form-error">{extractErrorMessage(mutation.error)}</div>}
      </Card>

      {run && (
        <Card title={`Trace ${run.trace_id.slice(0, 12)}… — ${formatDate(run.run_at)}`} className="mt-4">
          <div className="agent-chain">
            {AGENT_ORDER.map((name, i) => {
              const m = messagesByAgent.get(name);
              return (
                <div key={name} className="agent-chain-item">
                  <div className={`agent-step agent-step-${m?.status ?? "missing"}`}>
                    <div className="agent-step-title">{AGENT_LABEL[name]}</div>
                    <div className="agent-step-status">
                      {m ? (
                        <Badge tone={m.status === "ok" ? "success" : m.status === "warning" ? "warning" : "danger"}>
                          {m.status}
                        </Badge>
                      ) : <Badge tone="default">missing</Badge>}
                    </div>
                    {m?.confidence != null && <div className="agent-step-meta">confidence {m.confidence.toFixed(2)}</div>}
                    {m?.warnings?.length ? <div className="agent-step-warn">{m.warnings.length} warning(s)</div> : null}
                    {m?.errors?.length ? <div className="agent-step-error">{m.errors[0]}</div> : null}
                    <button className="btn btn-ghost btn-sm" onClick={() => setOpen(name)} disabled={!m}>
                      Inspect payload
                    </button>
                  </div>
                  {i < AGENT_ORDER.length - 1 && <div className="agent-arrow">↓</div>}
                </div>
              );
            })}
          </div>

          {open && messagesByAgent.get(open) && (
            <div className="json-panel">
              <div className="json-panel-title">{AGENT_LABEL[open]} — payload</div>
              <pre className="json-view">{JSON.stringify(messagesByAgent.get(open).payload, null, 2)}</pre>
              <div className="json-panel-title">Model versions</div>
              <pre className="json-view">{JSON.stringify(messagesByAgent.get(open).model_versions, null, 2)}</pre>
            </div>
          )}
        </Card>
      )}
    </>
  );
}