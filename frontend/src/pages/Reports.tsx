import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { listReports, getReport } from "@/services/reports";
import { Card } from "@/components/ui/Card";
import { PageHeader } from "@/components/ui/PageHeader";
import { Spinner } from "@/components/ui/Spinner";
import { Button } from "@/components/ui/Button";
import { Modal } from "@/components/ui/Modal";
import { API_BASE } from "@/services/api";

export default function Reports() {
  const [viewing, setViewing] = useState<string | null>(null);
  const list = useQuery({ queryKey: ["reports"], queryFn: listReports });
  const detail = useQuery({
    queryKey: ["report", viewing],
    queryFn: () => getReport(viewing!),
    enabled: !!viewing,
  });

  async function download(name: string) {
    const token = (await import("@/state/authStore")).useAuthStore.getState().accessToken;
    const url = `${API_BASE}/api/v1/reports/download/${encodeURIComponent(name)}`;
    const resp = await fetch(url, { headers: token ? { Authorization: `Bearer ${token}` } : {} });
    const blob = await resp.blob();
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = name;
    a.click();
  }

  return (
    <>
      <PageHeader title="Reports" subtitle="Metrics, monitoring output, and exported artefacts." />

      <Card>
        {list.isLoading ? <Spinner /> : (
          <table className="table">
            <thead>
              <tr>
                <th>Name</th>
                <th style={{ textAlign: "right" }}>Size (bytes)</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {(list.data ?? []).length === 0 && (
                <tr><td colSpan={3} className="table-empty">No reports yet. Run training / monitoring to generate them.</td></tr>
              )}
              {(list.data ?? []).map((r) => (
                <tr key={r.name}>
                  <td><code>{r.name}</code></td>
                  <td style={{ textAlign: "right" }}>{r.size_bytes.toLocaleString()}</td>
                  <td>
                    <Button size="sm" variant="ghost" onClick={() => setViewing(r.name)}>View</Button>{" "}
                    <Button size="sm" variant="secondary" onClick={() => download(r.name)}>Download</Button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Card>

      <Modal open={!!viewing} onClose={() => setViewing(null)} title={viewing ?? ""}>
        {detail.isLoading ? <Spinner /> : detail.data ? (
          <pre className="json-view" style={{ maxHeight: "60vh" }}>
            {typeof detail.data === "string" ? detail.data : JSON.stringify(detail.data, null, 2)}
          </pre>
        ) : null}
      </Modal>
    </>
  );
}