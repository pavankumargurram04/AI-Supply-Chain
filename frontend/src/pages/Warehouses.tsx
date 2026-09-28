import { useQuery } from "@tanstack/react-query";
import { listWarehouses } from "@/services/warehouses";
import { Card } from "@/components/ui/Card";
import { PageHeader } from "@/components/ui/PageHeader";
import { Spinner } from "@/components/ui/Spinner";
import { Badge } from "@/components/ui/Badge";
import { formatNumber } from "@/utils/format";

export default function Warehouses() {
  const wh = useQuery({ queryKey: ["warehouses"], queryFn: () => listWarehouses(200) });

  return (
    <>
      <PageHeader title="Warehouses" subtitle="Distribution centres and their capacity." />

      <Card>
        {wh.isLoading ? <Spinner /> : (
          <table className="table">
            <thead>
              <tr>
                <th>Code</th>
                <th>Name</th>
                <th>Region</th>
                <th>Country</th>
                <th style={{ textAlign: "right" }}>Capacity (units)</th>
              </tr>
            </thead>
            <tbody>
              {(wh.data ?? []).length === 0 && (
                <tr><td colSpan={5} className="table-empty">No warehouses.</td></tr>
              )}
              {(wh.data ?? []).map((w) => (
                <tr key={w.id}>
                  <td><code>{w.code}</code></td>
                  <td>{w.name}</td>
                  <td>{w.region ? <Badge tone="info">{w.region}</Badge> : "—"}</td>
                  <td>{w.country ?? "—"}</td>
                  <td style={{ textAlign: "right" }}>{formatNumber(w.capacity_units, 0)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Card>
    </>
  );
}