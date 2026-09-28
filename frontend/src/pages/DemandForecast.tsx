import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { listProducts } from "@/services/products";
import { listWarehouses } from "@/services/warehouses";
import { getForecast } from "@/services/forecast";
import { Card } from "@/components/ui/Card";
import { PageHeader } from "@/components/ui/PageHeader";
import { Spinner } from "@/components/ui/Spinner";
import { Button } from "@/components/ui/Button";
import { formatNumber } from "@/utils/format";

export default function DemandForecast() {
  const products = useQuery({ queryKey: ["products", "all"], queryFn: () => listProducts(undefined, 500) });
  const warehouses = useQuery({ queryKey: ["warehouses"], queryFn: () => listWarehouses(100) });
  const [sku, setSku] = useState<string>("");
  const [wh, setWh] = useState<string>("");
  const [triggered, setTriggered] = useState(false);

  const forecast = useQuery({
    queryKey: ["forecast", sku, wh, triggered],
    queryFn: () => getForecast(sku, wh),
    enabled: triggered && !!sku && !!wh,
  });

  const chartData = useMemo(() => {
    if (!forecast.data) return [];
    return forecast.data.point.map((p, i) => ({
      date: forecast.data!.forecast_dates[i],
      forecast: p,
      lower: forecast.data!.lower?.[i] ?? null,
      upper: forecast.data!.upper?.[i] ?? null,
    }));
  }, [forecast.data]);

  return (
    <>
      <PageHeader title="Demand Forecast" subtitle="Global LSTM. Uncertainty shown where quantile head is available." />

      <Card>
        <div className="toolbar">
          <label className="form-row form-row-inline">
            <span>Product</span>
            <select className="input" value={sku} onChange={(e) => setSku(e.target.value)}>
              <option value="">Select…</option>
              {products.data?.map((p) => <option key={p.id} value={p.sku}>{p.sku} — {p.name}</option>)}
            </select>
          </label>
          <label className="form-row form-row-inline">
            <span>Warehouse</span>
            <select className="input" value={wh} onChange={(e) => setWh(e.target.value)}>
              <option value="">Select…</option>
              {warehouses.data?.map((w) => <option key={w.id} value={w.code}>{w.code}</option>)}
            </select>
          </label>
          <Button onClick={() => setTriggered(true)} disabled={!sku || !wh}>Run forecast</Button>
        </div>

        {forecast.isLoading && <Spinner label="Running forecast…" />}
        {forecast.error && <div className="form-error">{(forecast.error as Error).message}</div>}

        {forecast.data && (
          <>
            <div className="muted-note">
              Origin {forecast.data.origin_date} · horizon {forecast.data.horizon} · model {forecast.data.model_version}
            </div>
            <ResponsiveContainer width="100%" height={360}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis />
                <Tooltip />
                <Line type="monotone" dataKey="forecast" stroke="#2563eb" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="lower" stroke="#94a3b8" strokeDasharray="3 3" dot={false} />
                <Line type="monotone" dataKey="upper" stroke="#94a3b8" strokeDasharray="3 3" dot={false} />
              </LineChart>
            </ResponsiveContainer>

            <div className="muted-note">
              Mean: {formatNumber(forecast.data.point.reduce((a, b) => a + b, 0) / forecast.data.point.length)} ·
              Sum: {formatNumber(forecast.data.point.reduce((a, b) => a + b, 0))}
            </div>
          </>
        )}
      </Card>
    </>
  );
}