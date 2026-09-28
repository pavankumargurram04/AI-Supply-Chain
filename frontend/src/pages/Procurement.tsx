import { useState } from "react";
import { useQuery, useQueryClient, useQuery as useQueryRaw } from "@tanstack/react-query";
import {
  listRecommendations,
  approve,
  reject,
  override,
  createRecommendation,
} from "@/services/procurement";
import { listProducts } from "@/services/products";
import { listWarehouses } from "@/services/warehouses";
import { extractErrorMessage } from "@/services/api";
import { Card } from "@/components/ui/Card";
import { PageHeader } from "@/components/ui/PageHeader";
import { Spinner } from "@/components/ui/Spinner";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Modal } from "@/components/ui/Modal";
import { formatCurrency, formatDate, formatNumber, formatPercent } from "@/utils/format";

export default function Procurement() {
  const qc = useQueryClient();
  const [statusFilter, setStatusFilter] = useState<string>("");
  const [overrideFor, setOverrideFor] = useState<number | null>(null);
  const [overrideQty, setOverrideQty] = useState<string>("");
  const [overrideReason, setOverrideReason] = useState<string>("");
  const [busy, setBusy] = useState<number | null>(null);

  // New recommendation state & queries
  const [createOpen, setCreateOpen] = useState(false);
  const [newSku, setNewSku] = useState("");
  const [newWh, setNewWh] = useState("");
  const [creating, setCreating] = useState(false);

  const products = useQueryRaw({
    queryKey: ["products", "for-procurement"],
    queryFn: () => listProducts(undefined, 500),
  });

  const warehouses = useQueryRaw({
    queryKey: ["warehouses", "for-procurement"],
    queryFn: () => listWarehouses(100),
  });

  const list = useQuery({
    queryKey: ["recommendations", statusFilter],
    queryFn: () => listRecommendations(statusFilter || undefined),
  });

  const rows = list.data ?? [];

  const productById = new Map(
    (products.data ?? []).map((p) => [p.id, p])
  );

  const warehouseById = new Map(
    (warehouses.data ?? []).map((w) => [w.id, w])
  );

  async function onApprove(id: number) {
    setBusy(id);
    try {
      await approve(id);
      await qc.invalidateQueries({ queryKey: ["recommendations"] });
      await qc.invalidateQueries({ queryKey: ["audit"] });
    } catch (err) {
      alert("Approve failed: " + extractErrorMessage(err));
    } finally {
      setBusy(null);
    }
  }

  async function onReject(id: number) {
    setBusy(id);
    try {
      await reject(id);
      await qc.invalidateQueries({ queryKey: ["recommendations"] });
      await qc.invalidateQueries({ queryKey: ["audit"] });
    } catch (err) {
      alert("Reject failed: " + extractErrorMessage(err));
    } finally {
      setBusy(null);
    }
  }

  async function submitOverride() {
    if (overrideFor === null) return;
    setBusy(overrideFor);
    try {
      await override(overrideFor, Number(overrideQty), overrideReason);
      setOverrideFor(null);
      setOverrideQty("");
      setOverrideReason("");
      await qc.invalidateQueries({ queryKey: ["recommendations"] });
      await qc.invalidateQueries({ queryKey: ["audit"] });
    } catch (err) {
      alert("Override failed: " + extractErrorMessage(err));
    } finally {
      setBusy(null);
    }
  }

  async function onCreateOrder(id: number) {
    setBusy(id);
    try {
      const { createOrderFromRecommendation } = await import("@/services/procurement");
      await createOrderFromRecommendation(id);
      await qc.invalidateQueries({ queryKey: ["recommendations"] });
      await qc.invalidateQueries({ queryKey: ["audit"] });
      await qc.invalidateQueries({ queryKey: ["shipments"] });
    } catch (err) {
      alert("Create PO failed: " + extractErrorMessage(err));
    } finally {
      setBusy(null);
    }
  }

  async function submitCreate() {
    if (!newSku || !newWh) return;
    setCreating(true);
    try {
      await createRecommendation({ product_sku: newSku, warehouse_code: newWh });
      setCreateOpen(false);
      setNewSku("");
      setNewWh("");
      await qc.invalidateQueries({ queryKey: ["recommendations"] });
      await qc.invalidateQueries({ queryKey: ["audit"] });
    } catch (err) {
      alert("Create recommendation failed: " + extractErrorMessage(err));
    } finally {
      setCreating(false);
    }
  }

  return (
    <>
      <PageHeader
        title="Procurement"
        subtitle="Recommendation queue with human approval workflow."
      />

      <Card>
        <div className="toolbar">
          <label className="form-row form-row-inline">
            <span>Status</span>
            <select
              className="input"
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
            >
              <option value="">All</option>
              <option value="pending_review">Pending review</option>
              <option value="approved">Approved</option>
              <option value="rejected">Rejected</option>
              <option value="overridden">Overridden</option>
              <option value="order_created">Order created</option>
            </select>
          </label>
          <Button variant="secondary" onClick={() => list.refetch()}>
            Refresh
          </Button>
          <Button variant="primary" onClick={() => setCreateOpen(true)}>
            New Recommendation
          </Button>
        </div>

        {list.isLoading ? (
          <Spinner />
        ) : (
          <table className="table">
            <thead>
              <tr>
                <th>Product</th>
                <th>Warehouse</th>
                <th>Trace</th>
                <th>Status</th>
                <th>Qty</th>
                <th>Cost</th>
                <th>Service</th>
                <th>Stockout</th>
                <th>Created</th>
                <th style={{ width: 260 }}>Actions</th>
              </tr>
            </thead>
            <tbody>
              {rows.length === 0 && (
                <tr>
                  <td colSpan={10} className="table-empty">
                    No recommendations.
                  </td>
                </tr>
              )}
              {rows.map((r) => (
                <tr key={r.id}>
                  <td>
                    {productById.get(r.product_id) ? (
                      <>
                        <code>{productById.get(r.product_id)!.sku}</code>
                        <div className="muted-note">
                          {productById.get(r.product_id)!.name}
                        </div>
                      </>
                    ) : (
                      `Product #${r.product_id}`
                    )}
                  </td>

                  <td>
                    {warehouseById.get(r.warehouse_id) ? (
                      <>
                        <code>{warehouseById.get(r.warehouse_id)!.code}</code>
                        <div className="muted-note">
                          {warehouseById.get(r.warehouse_id)!.name}
                        </div>
                      </>
                    ) : (
                      `Warehouse #${r.warehouse_id}`
                    )}
                  </td>

                  <td>
                    <code>{r.trace_id.slice(0, 12)}…</code>
                  </td>
                  <td>
                    <Badge
                      tone={
                        r.status === "approved"
                          ? "success"
                          : r.status === "rejected"
                          ? "danger"
                          : r.status === "overridden"
                          ? "warning"
                          : "info"
                      }
                    >
                      {r.status}
                    </Badge>
                  </td>
                  <td>{formatNumber(r.final_quantity)}</td>
                  <td>{formatCurrency(r.expected_cost)}</td>
                  <td>{formatPercent(r.expected_service_level)}</td>
                  <td>{formatPercent(r.stockout_probability)}</td>
                  <td>{formatDate(r.created_at)}</td>
                  <td>
                    {(r.status === "pending_review" || r.status === "generated") && (
                      <>
                        <Button size="sm" onClick={() => onApprove(r.id)} loading={busy === r.id}>
                          Approve
                        </Button>{" "}
                        <Button
                          size="sm"
                          variant="secondary"
                          onClick={() => onReject(r.id)}
                          loading={busy === r.id}
                        >
                          Reject
                        </Button>{" "}
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={() => {
                            setOverrideFor(r.id);
                            setOverrideQty(String(r.final_quantity ?? ""));
                          }}
                        >
                          Override
                        </Button>
                      </>
                    )}
                    {(r.status === "approved" || r.status === "overridden") && (
                      <Button
                        size="sm"
                        variant="primary"
                        onClick={() => onCreateOrder(r.id)}
                        loading={busy === r.id}
                      >
                        Create PO
                      </Button>
                    )}
                    {r.status === "order_created" && <Badge tone="success">PO created</Badge>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Card>

      <Modal
        open={overrideFor !== null}
        onClose={() => setOverrideFor(null)}
        title="Override recommendation"
        footer={
          <>
            <Button variant="secondary" onClick={() => setOverrideFor(null)}>
              Cancel
            </Button>
            <Button onClick={submitOverride} disabled={!overrideQty || overrideReason.length < 3}>
              Submit override
            </Button>
          </>
        }
      >
        <label className="form-row">
          <span>New quantity</span>
          <input
            className="input"
            type="number"
            value={overrideQty}
            onChange={(e) => setOverrideQty(e.target.value)}
          />
        </label>
        <label className="form-row">
          <span>Reason (min 3 characters)</span>
          <textarea
            className="input"
            rows={3}
            value={overrideReason}
            onChange={(e) => setOverrideReason(e.target.value)}
          />
        </label>
      </Modal>

      <Modal
        open={createOpen}
        onClose={() => setCreateOpen(false)}
        title="Create a new recommendation"
        footer={
          <>
            <Button variant="secondary" onClick={() => setCreateOpen(false)}>
              Cancel
            </Button>
            <Button onClick={submitCreate} loading={creating} disabled={!newSku || !newWh}>
              Run recommendation
            </Button>
          </>
        }
      >
        <label className="form-row">
          <span>Product</span>
          <select
            className="input"
            value={newSku}
            onChange={(e) => setNewSku(e.target.value)}
          >
            <option value="">Select a product…</option>
            {products.data?.map((p) => (
              <option key={p.id} value={p.sku}>
                {p.sku} — {p.name}
              </option>
            ))}
          </select>
        </label>
        <label className="form-row">
          <span>Warehouse</span>
          <select
            className="input"
            value={newWh}
            onChange={(e) => setNewWh(e.target.value)}
          >
            <option value="">Select a warehouse…</option>
            {warehouses.data?.map((w) => (
              <option key={w.id} value={w.code}>
                {w.code} — {w.name}
              </option>
            ))}
          </select>
        </label>
        <div className="muted-note">
          This runs the full decision engine on the server and persists the result. Requires trained
          models under <code>artifacts/demand/dev/</code>, etc.
        </div>
      </Modal>
    </>
  );
}