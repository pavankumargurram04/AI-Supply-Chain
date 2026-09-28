import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { listProducts } from "@/services/products";
import { listWarehouses } from "@/services/warehouses";
import { getInventoryState, recommend } from "@/services/inventory";
import { Card } from "@/components/ui/Card";
import { PageHeader } from "@/components/ui/PageHeader";
import { Spinner } from "@/components/ui/Spinner";
import { Button } from "@/components/ui/Button";
import { Kpi } from "@/components/ui/Kpi";
import { formatCurrency, formatNumber, formatPercent } from "@/utils/format";

export default function Inventory() {
  const products = useQuery({ queryKey: ["products", "all"], queryFn: () => listProducts(undefined, 500) });
  const warehouses = useQuery({ queryKey: ["warehouses"], queryFn: () => listWarehouses(100) });

  const [sku, setSku] = useState("");
  const [wh, setWh] = useState("");
  const [triggered, setTriggered] = useState(false);

  // Scenario knobs
  const [demandMult, setDemandMult] = useState(1.0);
  const [leadTimeOverride, setLeadTimeOverride] = useState<string>("");
  const [inventoryOverride, setInventoryOverride] = useState<string>("");

  const inventory = useQuery({
    queryKey: ["inventory-state", sku, wh, triggered],
    queryFn: () => getInventoryState(sku, wh),
    enabled: triggered && !!sku && !!wh,
  });

  const recommendation = useQuery({
    queryKey: ["inventory-recommend", sku, wh, demandMult, leadTimeOverride, inventoryOverride, triggered],
    queryFn: () => recommend({
      product_sku: sku,
      warehouse_code: wh,
      demand_multiplier: demandMult,
      lead_time_override_days: leadTimeOverride ? Number(leadTimeOverride) : null,
      inventory_override: inventoryOverride ? Number(inventoryOverride) : null,
    }),
    enabled: triggered && !!sku && !!wh,
  });

  const hasAny = useMemo(() => !!inventory.data || !!recommendation.data, [inventory.data, recommendation.data]);

  return (
    <>
      <PageHeader title="Inventory" subtitle="Inventory state, scenario knobs, and simulation-based recommendation." />

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
          <Button onClick={() => setTriggered(true)} disabled={!sku || !wh}>Load</Button>
        </div>

        {triggered && (
          <div className="toolbar">
            <label className="form-row form-row-inline">
              <span>Demand ×</span>
              <input className="input" type="number" step="0.1" value={demandMult}
                     onChange={(e) => setDemandMult(Number(e.target.value))} />
            </label>
            <label className="form-row form-row-inline">
              <span>Lead time override (days)</span>
              <input className="input" type="number" placeholder="—" value={leadTimeOverride}
                     onChange={(e) => setLeadTimeOverride(e.target.value)} />
            </label>
            <label className="form-row form-row-inline">
              <span>Inventory override</span>
              <input className="input" type="number" placeholder="—" value={inventoryOverride}
                     onChange={(e) => setInventoryOverride(e.target.value)} />
            </label>
          </div>
        )}
      </Card>

      {inventory.isLoading && <Spinner label="Loading inventory…" />}
      {inventory.data && (
        <Card title="Current state" className="mt-4">
          <div className="grid grid-cols-5">
            <Kpi label="On hand" value={formatNumber(inventory.data.on_hand)} />
            <Kpi label="Reserved" value={formatNumber(inventory.data.reserved)} />
            <Kpi label="Inbound" value={formatNumber(inventory.data.inbound)} />
            <Kpi label="Available" value={formatNumber(inventory.data.available)} />
            <Kpi label="Position" value={formatNumber(inventory.data.inventory_position)} />
          </div>
        </Card>
      )}

      {recommendation.isLoading && <Spinner label="Running simulation…" />}
      {recommendation.error && <div className="form-error">{(recommendation.error as Error).message}</div>}
      {recommendation.data && (
        <Card title="Recommendation" className="mt-4">
          <div className="grid grid-cols-4">
            <Kpi label="Recommended qty" value={formatNumber(recommendation.data.final_quantity)} />
            <Kpi label="Total cost" value={formatCurrency(recommendation.data.total_cost)} />
            <Kpi label="Expected service" value={formatPercent(recommendation.data.expected_service_level)}
                 tone={recommendation.data.expected_service_level > 0.9 ? "good" : "warn"} />
            <Kpi label="Stockout probability" value={formatPercent(recommendation.data.stockout_probability)}
                 tone={recommendation.data.stockout_probability > 0.3 ? "bad" : "good"} />
          </div>
          <div className="grid grid-cols-3" style={{ marginTop: 16 }}>
            <Kpi label="Safety stock" value={formatNumber(recommendation.data.safety_stock)} />
            <Kpi label="Reorder point" value={formatNumber(recommendation.data.reorder_point)} />
            <Kpi label="Confidence" value={formatPercent(recommendation.data.confidence)} />
          </div>
          <div className="explanation-block">
            <div className="explanation-title">Explanation</div>
            <pre className="explanation-text">{recommendation.data.explanation}</pre>
          </div>
        </Card>
      )}

      {!hasAny && !triggered && (
        <Card className="mt-4">
          <div className="muted-note">Select a product and warehouse to begin.</div>
        </Card>
      )}
    </>
  );
}