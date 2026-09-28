import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { listShipments, getShipment } from "@/services/shipments";
import { Card } from "@/components/ui/Card";
import { PageHeader } from "@/components/ui/PageHeader";
import { Spinner } from "@/components/ui/Spinner";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Modal } from "@/components/ui/Modal";
import { formatDateShort, formatDate } from "@/utils/format";

export default function Shipments() {
  const [status, setStatus] = useState("");
  const [selected, setSelected] = useState<number | null>(null);

  const list = useQuery({
    queryKey: ["shipments", status],
    queryFn: () => listShipments(status || undefined),
  });

  const detail = useQuery({
    queryKey: ["shipment", selected],
    queryFn: () => getShipment(selected!),
    enabled: selected !== null,
  });

  return (
    <>
      <PageHeader title="Shipments" subtitle="Purchase-order shipments and their event timeline." />

      <Card>
        <div className="toolbar">
          <label className="form-row form-row-inline">
            <span>Status</span>
            <select className="input" value={status} onChange={(e) => setStatus(e.target.value)}>
              <option value="">All</option>
              <option value="planned">Planned</option>
              <option value="in_transit">In transit</option>
              <option value="delayed">Delayed</option>
              <option value="arrived">Arrived</option>
              <option value="cancelled">Cancelled</option>
            </select>
          </label>
          <Button variant="secondary" onClick={() => list.refetch()}>Refresh</Button>
        </div>

        {list.isLoading ? <Spinner /> : (
          <table className="table">
            <thead>
              <tr>
                <th>ID</th>
                <th>PO</th>
                <th>Carrier</th>
                <th>Tracking</th>
                <th>Status</th>
                <th>Expected</th>
                <th>Arrived</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {(list.data ?? []).length === 0 && (
                <tr><td colSpan={8} className="table-empty">No shipments.</td></tr>
              )}
              {(list.data ?? []).map((s) => (
                <tr key={s.id}>
                  <td>#{s.id}</td>
                  <td>{s.po_number}</td>
                  <td>{s.carrier ?? "—"}</td>
                  <td><code>{s.tracking_number ?? "—"}</code></td>
                  <td><Badge tone={s.status === "arrived" ? "success" : s.status === "delayed" ? "danger" : "info"}>{s.status}</Badge></td>
                  <td>{formatDateShort(s.expected_arrival)}</td>
                  <td>{formatDateShort(s.actual_arrival)}</td>
                  <td><Button size="sm" variant="ghost" onClick={() => setSelected(s.id)}>View</Button></td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Card>

      <Modal open={selected !== null} onClose={() => setSelected(null)} title={`Shipment #${selected ?? ""}`}>
        {detail.isLoading ? <Spinner /> : detail.data ? (
          <div>
            <div className="kv-grid">
              <div><span>Status</span><b>{detail.data.status}</b></div>
              <div><span>Carrier</span><b>{detail.data.carrier ?? "—"}</b></div>
              <div><span>Tracking</span><b>{detail.data.tracking_number ?? "—"}</b></div>
              <div><span>Expected</span><b>{formatDateShort(detail.data.expected_arrival)}</b></div>
              <div><span>Arrived</span><b>{formatDateShort(detail.data.actual_arrival)}</b></div>
            </div>
            <div className="timeline-title">Event timeline</div>
            <ul className="timeline">
              {(detail.data.events ?? []).map((e: any, i: number) => (
                <li key={i}>
                  <div className="timeline-dot" />
                  <div>
                    <div className="timeline-action">{e.event_type}</div>
                    <div className="timeline-meta">{formatDate(e.occurred_at)} {e.location && `· ${e.location}`}</div>
                    {e.notes && <div className="timeline-notes">{e.notes}</div>}
                  </div>
                </li>
              ))}
            </ul>
          </div>
        ) : null}
      </Modal>
    </>
  );
}