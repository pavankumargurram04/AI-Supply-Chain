import { useMemo, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { listProducts } from "@/services/products";
import { listWarehouses } from "@/services/warehouses";
import { getForecast } from "@/services/forecast";
import { getSupplierRisk } from "@/services/supplierRisk";
import { getInventoryState, recommend } from "@/services/inventory";
import { Card } from "@/components/ui/Card";
import { PageHeader } from "@/components/ui/PageHeader";
import { Spinner } from "@/components/ui/Spinner";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Kpi } from "@/components/ui/Kpi";
import { formatCurrency, formatNumber, formatPercent } from "@/utils/format";
import {
  Area,
  CartesianGrid,
  Line,
  AreaChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

type Tab = "forecast" | "suppliers" | "inventory" | "recommendation";

export default function ProductDetail() {
  const { sku } = useParams<{ sku: string }>();
  const navigate = useNavigate();

  const [tab, setTab] = useState<Tab>("forecast");
  const [warehouse, setWarehouse] = useState<string | null>(null);

  const products = useQuery({
    queryKey: ["products", "all"],
    queryFn: () => listProducts(undefined, 500),
  });

  const warehouses = useQuery({
    queryKey: ["warehouses"],
    queryFn: () => listWarehouses(100),
  });

  const product = useMemo(
    () => products.data?.find((p) => p.sku === sku),
    [products.data, sku]
  );

  const currentWh =
    warehouse ?? warehouses.data?.[0]?.code ?? null;

  const forecast = useQuery({
    queryKey: ["forecast", sku, currentWh],
    queryFn: () => getForecast(sku!, currentWh!),
    enabled: !!sku && !!currentWh,
  });

  const suppliers = useQuery({
    queryKey: ["supplier-risk", sku],
    queryFn: () => getSupplierRisk(sku!),
    enabled: !!sku,
  });

  const inventory = useQuery({
    queryKey: ["inventory", sku, currentWh],
    queryFn: () => getInventoryState(sku!, currentWh!),
    enabled: !!sku && !!currentWh,
  });

  const recommendation = useQuery({
    queryKey: ["recommendation", sku, currentWh, tab],
    queryFn: () =>
      recommend({
        product_sku: sku!,
        warehouse_code: currentWh!,
      }),
    enabled:
      tab === "recommendation" &&
      !!sku &&
      !!currentWh,
    staleTime: 0,
  });

  if (products.isLoading) {
    return <Spinner label="Loading product…" />;
  }

  if (!product) {
    return (
      <Card>
        <div className="form-error">
          Product {sku ?? "unknown"} was not found.
        </div>

        <Button
          variant="secondary"
          onClick={() => navigate("/products")}
          style={{ marginTop: 12 }}
        >
          Back to Products
        </Button>
      </Card>
    );
  }

  return (
    <>
      <PageHeader
        title={`${product.sku} — ${product.name}`}
        subtitle="Product intelligence: forecast, supplier risk, inventory state, and recommendation."
        actions={
          <Button
            variant="secondary"
            onClick={() => navigate("/products")}
          >
            Back
          </Button>
        }
      />

      {/* Product master information */}
      <Card title="Product information">
        <div className="grid grid-cols-4">
          <Kpi
            label="Unit cost"
            value={formatCurrency(product.unit_cost)}
          />

          <Kpi
            label="Unit price"
            value={formatCurrency(product.unit_price)}
          />

          <Kpi
            label="Base demand"
            value={formatNumber(product.base_demand, 1)}
          />

          <Kpi
            label="Lead time"
            value={`${product.lead_time_days} d`}
          />
        </div>

        <div
          className="grid grid-cols-4"
          style={{ marginTop: 16 }}
        >
          <Kpi
            label="Weight"
            value={
              product.weight_kg != null
                ? `${formatNumber(product.weight_kg, 3)} kg`
                : "—"
            }
          />

          <Kpi
            label="Volume"
            value={
              product.volume_m3 != null
                ? `${formatNumber(product.volume_m3, 6)} m³`
                : "—"
            }
          />

          <div>
            <div className="muted-note">Lifecycle</div>
            <Badge
              tone={
                product.lifecycle_stage === "maturity"
                  ? "success"
                  : "info"
              }
            >
              {product.lifecycle_stage}
            </Badge>
          </div>

          <div>
            <div className="muted-note">Category ID</div>
            <strong>
              {product.category_id ?? "—"}
            </strong>
          </div>
        </div>

        <div
          className="grid grid-cols-3"
          style={{ marginTop: 16 }}
        >
          <div>
            <div className="muted-note">Seasonality</div>
            <Badge tone="info">
              {product.seasonality_class}
            </Badge>
          </div>

          <div>
            <div className="muted-note">Volatility</div>
            <Badge tone="info">
              {product.volatility_class}
            </Badge>
          </div>

          <div>
            <div className="muted-note">Description</div>
            <div>
              {product.description ?? "—"}
            </div>
          </div>
        </div>
      </Card>

      <div
        className="toolbar"
        style={{ marginTop: 16 }}
      >
        <label className="form-row form-row-inline">
          <span>Warehouse</span>

          <select
            className="input"
            value={currentWh ?? ""}
            onChange={(e) =>
              setWarehouse(e.target.value)
            }
          >
            {warehouses.data?.map((w) => (
              <option
                key={w.id}
                value={w.code}
              >
                {w.code} — {w.name}
              </option>
            ))}
          </select>
        </label>
      </div>

      <div className="tabs">
        {(
          [
            "forecast",
            "suppliers",
            "inventory",
            "recommendation",
          ] as Tab[]
        ).map((t) => (
          <button
            key={t}
            className={
              tab === t
                ? "tab tab-active"
                : "tab"
            }
            onClick={() => setTab(t)}
          >
            {t === "forecast"
              ? "Forecast"
              : t === "suppliers"
              ? "Suppliers"
              : t === "inventory"
              ? "Inventory"
              : "Recommendation"}
          </button>
        ))}
      </div>

      {tab === "forecast" && (
        <Card title="Demand forecast">
          {forecast.isLoading ? (
            <Spinner />
          ) : forecast.error ? (
            <div className="form-error">
              {(forecast.error as Error).message}
            </div>
          ) : forecast.data ? (
            <>
              <div className="muted-note">
                Model: {forecast.data.model_version} ·
                feature: {forecast.data.feature_version}
              </div>

              <ResponsiveContainer
                width="100%"
                height={280}
              >
                <AreaChart
                  data={forecast.data.point.map(
                    (p, i) => ({
                      date:
                        forecast.data!
                          .forecast_dates[i],
                      point: p,
                      lower:
                        forecast.data!.lower?.[i] ??
                        p,
                      upper:
                        forecast.data!.upper?.[i] ??
                        p,
                    })
                  )}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis />
                  <Tooltip />

                  <Area
                    type="monotone"
                    dataKey="upper"
                    stroke="none"
                    fill="#bfdbfe"
                  />

                  <Area
                    type="monotone"
                    dataKey="lower"
                    stroke="none"
                    fill="#ffffff"
                  />

                  <Line
                    type="monotone"
                    dataKey="point"
                    stroke="#2563eb"
                    strokeWidth={2}
                    dot={false}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </>
          ) : null}
        </Card>
      )}

      {tab === "suppliers" && (
        <Card title="Supplier candidates & risk">
          {suppliers.isLoading ? (
            <Spinner />
          ) : suppliers.data &&
            suppliers.data.length > 0 ? (
            <table className="table">
              <thead>
                <tr>
                  <th>Supplier</th>
                  <th>Risk</th>
                  <th>Label</th>
                  <th>Lead time</th>
                  <th>Unit price</th>
                  <th>MOQ</th>
                </tr>
              </thead>

              <tbody>
                {suppliers.data.map((s) => (
                  <tr key={s.supplier_code}>
                    <td>
                      <code>
                        {s.supplier_code}
                      </code>
                    </td>

                    <td>
                      <div className="risk-bar">
                        <div
                          className="risk-fill"
                          style={{
                            width: `${Math.round(
                              s.risk_probability * 100
                            )}%`,
                          }}
                        />
                      </div>

                      {formatPercent(
                        s.risk_probability
                      )}
                    </td>

                    <td>
                      <Badge
                        tone={
                          s.risk_label === "high"
                            ? "danger"
                            : "success"
                        }
                      >
                        {s.risk_label}
                      </Badge>
                    </td>

                    <td>
                      {formatNumber(
                        s.lead_time_days,
                        1
                      )}{" "}
                      d
                    </td>

                    <td>
                      {formatCurrency(
                        s.unit_price
                      )}
                    </td>

                    <td>{s.moq}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <div className="muted-note">
              No supplier candidates.
            </div>
          )}
        </Card>
      )}

      {tab === "inventory" && (
        <Card title="Inventory state">
          {inventory.isLoading ? (
            <Spinner />
          ) : inventory.error ? (
            <div className="form-error">
              {(inventory.error as Error).message}
            </div>
          ) : inventory.data ? (
            <div className="grid grid-cols-4">
              <Kpi
                label="On hand"
                value={formatNumber(
                  inventory.data.on_hand
                )}
              />

              <Kpi
                label="Reserved"
                value={formatNumber(
                  inventory.data.reserved
                )}
              />

              <Kpi
                label="Inbound"
                value={formatNumber(
                  inventory.data.inbound
                )}
              />

              <Kpi
                label="Available"
                value={formatNumber(
                  inventory.data.available
                )}
              />
            </div>
          ) : null}
        </Card>
      )}

      {tab === "recommendation" && (
        <Card title="Recommendation">
          {recommendation.isLoading ? (
            <Spinner label="Running decision engine…" />
          ) : recommendation.error ? (
            <div className="form-error">
              {(recommendation.error as Error).message}
            </div>
          ) : recommendation.data ? (
            <>
              <div className="grid grid-cols-4">
                <Kpi
                  label="Recommended qty"
                  value={formatNumber(
                    recommendation.data.final_quantity
                  )}
                />

                <Kpi
                  label="Expected cost"
                  value={formatCurrency(
                    recommendation.data.total_cost
                  )}
                />

                <Kpi
                  label="Service level"
                  value={formatPercent(
                    recommendation.data
                      .expected_service_level
                  )}
                  tone={
                    recommendation.data
                      .expected_service_level > 0.9
                      ? "good"
                      : "warn"
                  }
                />

                <Kpi
                  label="Stockout prob."
                  value={formatPercent(
                    recommendation.data
                      .stockout_probability
                  )}
                  tone={
                    recommendation.data
                      .stockout_probability > 0.3
                      ? "bad"
                      : "good"
                  }
                />
              </div>

              <div
                className="grid grid-cols-3"
                style={{ marginTop: 16 }}
              >
                <Kpi
                  label="Safety stock"
                  value={formatNumber(
                    recommendation.data
                      .safety_stock
                  )}
                />

                <Kpi
                  label="Reorder point"
                  value={formatNumber(
                    recommendation.data
                      .reorder_point
                  )}
                />

                <Kpi
                  label="Confidence"
                  value={formatPercent(
                    recommendation.data.confidence
                  )}
                />
              </div>

              {recommendation.data
                .guardrail_violations.length > 0 && (
                <div
                  className="form-error"
                  style={{ marginTop: 12 }}
                >
                  Guardrail violations:{" "}
                  {recommendation.data.guardrail_violations.join(
                    "; "
                  )}
                </div>
              )}

              <div className="explanation-block">
                <div className="explanation-title">
                  Explanation (numeric evidence)
                </div>

                <pre className="explanation-text">
                  {recommendation.data.explanation}
                </pre>
              </div>
            </>
          ) : null}
        </Card>
      )}
    </>
  );
}