import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { listProducts } from "@/services/products";
import { listWarehouses } from "@/services/warehouses";
import { runScenario, type ScenarioResult } from "@/services/scenarios";
import { useQuery } from "@tanstack/react-query";
import { Card } from "@/components/ui/Card";
import { PageHeader } from "@/components/ui/PageHeader";
import { Button } from "@/components/ui/Button";
import { Kpi } from "@/components/ui/Kpi";
import { extractErrorMessage } from "@/services/api";
import { formatCurrency, formatNumber, formatPercent } from "@/utils/format";

type ScenarioType = "demand_spike" | "lead_time_increase" | "inventory_level" | "budget_constraint";

export default function ScenarioLab() {
  const products = useQuery({ queryKey: ["products", "all"], queryFn: () => listProducts(undefined, 500) });
  const warehouses = useQuery({ queryKey: ["warehouses"], queryFn: () => listWarehouses(100) });

  const [sku, setSku] = useState("");
  const [wh, setWh] = useState("");
  const [scenarioType, setScenarioType] = useState<ScenarioType>("demand_spike");
  const [demandMult, setDemandMult] = useState(1.5);
  const [leadTime, setLeadTime] = useState("");
  const [inventoryOverride, setInventoryOverride] = useState("");
  const [budget, setBudget] = useState("");
  const [result, setResult] = useState<ScenarioResult | null>(null);

  const mutation = useMutation({
    mutationFn: () =>
      runScenario({
        product_sku: sku,
        warehouse_code: wh,
        scenario_type: scenarioType,
        demand_multiplier: demandMult,
        lead_time_override_days: leadTime ? Number(leadTime) : null,
        inventory_override: inventoryOverride ? Number(inventoryOverride) : null,
        budget: budget ? Number(budget) : null,
      }),
    onSuccess: (r) => setResult(r),
  });

  const compareData = result
    ? [
        { name: "Order qty", baseline: result.baseline.final_quantity, scenario: result.scenario.final_quantity },
        { name: "Total cost", baseline: result.baseline.total_cost, scenario: result.scenario.total_cost },
        { name: "Safety stock", baseline: result.baseline.safety_stock, scenario: result.scenario.safety_stock },
        { name: "Reorder point", baseline: result.baseline.reorder_point, scenario: result.scenario.reorder_point },
      ]
    : [];

  return (
    <>
      <PageHeader title="Scenario Lab" subtitle="Compare a baseline plan against a scenario using the real decision engine." />

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
          <label className="form-row form-row-inline">
            <span>Scenario</span>
            <select className="input" value={scenarioType} onChange={(e) => setScenarioType(e.target.value as ScenarioType)}>
              <option value="demand_spike">Demand spike</option>
              <option value="lead_time_increase">Lead time increase</option>
              <option value="inventory_level">Inventory override</option>
              <option value="budget_constraint">Budget constraint</option>
            </select>
          </label>
        </div>

        <div className="toolbar">
          {(scenarioType === "demand_spike") && (
            <label className="form-row form-row-inline">
              <span>Demand ×</span>
              <input className="input" type="number" step="0.1" min="0" value={demandMult}
                     onChange={(e) => setDemandMult(Number(e.target.value))} />
            </label>
          )}
          {(scenarioType === "lead_time_increase") && (
            <label className="form-row form-row-inline">
              <span>Lead time (days)</span>
              <input className="input" type="number" value={leadTime} onChange={(e) => setLeadTime(e.target.value)} />
            </label>
          )}
          {(scenarioType === "inventory_level") && (
            <label className="form-row form-row-inline">
              <span>Inventory override</span>
              <input className="input" type="number" value={inventoryOverride} onChange={(e) => setInventoryOverride(e.target.value)} />
            </label>
          )}
          {(scenarioType === "budget_constraint") && (
            <label className="form-row form-row-inline">
              <span>Budget</span>
              <input className="input" type="number" value={budget} onChange={(e) => setBudget(e.target.value)} />
            </label>
          )}
          <Button onClick={() => mutation.mutate()} loading={mutation.isPending} disabled={!sku || !wh}>
            Run scenario
          </Button>
        </div>

        {mutation.error && <div className="form-error">{extractErrorMessage(mutation.error)}</div>}
      </Card>

      {result && (
        <>
          <Card title="Baseline vs. scenario" className="mt-4">
            <div className="grid grid-cols-4">
              <Kpi label="Δ Order quantity" value={formatNumber(result.delta.final_quantity, 1)} tone={result.delta.final_quantity > 0 ? "warn" : "good"} />
              <Kpi label="Δ Total cost" value={formatCurrency(result.delta.total_cost)} />
              <Kpi label="Δ Service level" value={formatPercent(result.delta.expected_service_level)} tone={result.delta.expected_service_level < 0 ? "bad" : "good"} />
              <Kpi label="Δ Stockout prob." value={formatPercent(result.delta.stockout_probability)} tone={result.delta.stockout_probability > 0 ? "bad" : "good"} />
            </div>
          </Card>

          <Card title="Side-by-side comparison" className="mt-4">
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={compareData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="baseline" fill="#94a3b8" />
                <Bar dataKey="scenario" fill="#2563eb" />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Numeric detail" className="mt-4">
            <table className="table">
              <thead>
                <tr>
                  <th>Metric</th>
                  <th style={{ textAlign: "right" }}>Baseline</th>
                  <th style={{ textAlign: "right" }}>Scenario</th>
                  <th style={{ textAlign: "right" }}>Δ</th>
                </tr>
              </thead>
              <tbody>
                {[
                  ["Order quantity", result.baseline.final_quantity, result.scenario.final_quantity],
                  ["Total cost", result.baseline.total_cost, result.scenario.total_cost],
                  ["Service level", result.baseline.expected_service_level, result.scenario.expected_service_level],
                  ["Stockout probability", result.baseline.stockout_probability, result.scenario.stockout_probability],
                  ["Safety stock", result.baseline.safety_stock, result.scenario.safety_stock],
                  ["Reorder point", result.baseline.reorder_point, result.scenario.reorder_point],
                ].map(([label, b, s]) => (
                  <tr key={String(label)}>
                    <td>{label}</td>
                    <td style={{ textAlign: "right" }}>{formatNumber(Number(b), 3)}</td>
                    <td style={{ textAlign: "right" }}>{formatNumber(Number(s), 3)}</td>
                    <td style={{ textAlign: "right", color: Number(s) > Number(b) ? "#b91c1c" : "#166534" }}>
                      {formatNumber(Number(s) - Number(b), 3)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>
        </>
      )}
    </>
  );
}