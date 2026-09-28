import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { listProducts } from "@/services/products";
import { getSupplierRisk } from "@/services/supplierRisk";
import { Card } from "@/components/ui/Card";
import { PageHeader } from "@/components/ui/PageHeader";
import { Spinner } from "@/components/ui/Spinner";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { formatCurrency, formatNumber, formatPercent } from "@/utils/format";

export default function SupplierRisk() {
  const products = useQuery({ queryKey: ["products", "all"], queryFn: () => listProducts(undefined, 500) });
  const [sku, setSku] = useState("");
  const [submitted, setSubmitted] = useState("");

  const risk = useQuery({
    queryKey: ["supplier-risk", submitted],
    queryFn: () => getSupplierRisk(submitted),
    enabled: !!submitted,
  });

  const rows = risk.data ?? [];

  return (
    <>
      <PageHeader title="Supplier Risk" subtitle="Per-supplier disruption probability from the XGBoost classifier." />

      <Card>
        <div className="toolbar">
          <label className="form-row form-row-inline">
            <span>Product</span>
            <select className="input" value={sku} onChange={(e) => setSku(e.target.value)}>
              <option value="">Select…</option>
              {products.data?.map((p) => <option key={p.id} value={p.sku}>{p.sku} — {p.name}</option>)}
            </select>
          </label>
          <Button onClick={() => setSubmitted(sku)} disabled={!sku}>Score suppliers</Button>
        </div>

        {risk.isLoading && <Spinner label="Scoring…" />}
        {risk.error && <div className="form-error">{(risk.error as Error).message}</div>}

        {rows.length > 0 && (
          <table className="table">
            <thead>
              <tr>
                <th>Supplier</th>
                <th>Risk probability</th>
                <th>Label</th>
                <th>Lead time</th>
                <th>Unit price</th>
                <th>MOQ</th>
                <th>Preferred</th>
                <th>Model version</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.supplier_code}>
                  <td><code>{r.supplier_code}</code></td>
                  <td style={{ minWidth: 180 }}>
                    <div className="risk-bar"><div className="risk-fill" style={{ width: `${Math.round(r.risk_probability * 100)}%` }} /></div>
                    {formatPercent(r.risk_probability)}
                  </td>
                  <td><Badge tone={r.risk_label === "high" ? "danger" : "success"}>{r.risk_label}</Badge></td>
                  <td>{formatNumber(r.lead_time_days, 1)} d</td>
                  <td>{formatCurrency(r.unit_price)}</td>
                  <td>{r.moq}</td>
<td>
  <Badge tone={r.is_preferred ? "success" : "info"}>
    {r.is_preferred ? "Yes" : "No"}
  </Badge>
</td>
<td><code>{r.model_version ?? "—"}</code></td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Card>
    </>
  );
}