import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { listProducts } from "@/services/products";
import { Card } from "@/components/ui/Card";
import { PageHeader } from "@/components/ui/PageHeader";
import { Table } from "@/components/ui/Table";
import { Spinner } from "@/components/ui/Spinner";
import { Badge } from "@/components/ui/Badge";
import { formatCurrency, formatNumber } from "@/utils/format";

export default function Products() {
  const [search, setSearch] = useState("");
  const [submitted, setSubmitted] = useState("");
  const navigate = useNavigate();

  const { data, isLoading, isFetching } = useQuery({
    queryKey: ["products", submitted],
    queryFn: () => listProducts(submitted || undefined, 200),
  });

  const rows = useMemo(() => data ?? [], [data]);

  return (
    <>
      <PageHeader
        title="Products"
        subtitle="Catalogue of SKUs. Search dynamically against the database."
      />

      <Card>
        <div className="toolbar">
          <input
            className="input"
            placeholder="Search by SKU or name..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                setSubmitted(search);
              }
            }}
          />

          <button
            className="btn btn-primary btn-md"
            onClick={() => setSubmitted(search)}
          >
            Search
          </button>

          {submitted && (
            <button
              className="btn btn-ghost btn-md"
              onClick={() => {
                setSearch("");
                setSubmitted("");
              }}
            >
              Clear
            </button>
          )}
        </div>

        {isLoading ? (
          <Spinner />
        ) : (
          <Table
            rows={rows}
            rowKey={(r) => r.id}
            onRowClick={(r) => navigate(`/products/${r.sku}`)}
            columns={[
              {
                key: "sku",
                header: "SKU",
                render: (r) => <code>{r.sku}</code>,
              },

              {
                key: "name",
                header: "Name",
                render: (r) => r.name,
              },

              {
                key: "description",
                header: "Description",
                render: (r) => r.description ?? "—",
              },

              {
                key: "category",
                header: "Category ID",
                align: "right",
                render: (r) => r.category_id ?? "—",
              },

              {
                key: "cost",
                header: "Unit cost",
                align: "right",
                render: (r) => formatCurrency(r.unit_cost),
              },

              {
                key: "price",
                header: "Unit price",
                align: "right",
                render: (r) => formatCurrency(r.unit_price),
              },

              {
                key: "weight",
                header: "Weight",
                align: "right",
                render: (r) =>
                  r.weight_kg != null
                    ? `${formatNumber(r.weight_kg, 3)} kg`
                    : "—",
              },

              {
                key: "volume",
                header: "Volume",
                align: "right",
                render: (r) =>
                  r.volume_m3 != null
                    ? `${formatNumber(r.volume_m3, 6)} m³`
                    : "—",
              },

              {
                key: "base",
                header: "Base demand",
                align: "right",
                render: (r) => formatNumber(r.base_demand, 1),
              },

              {
                key: "lead",
                header: "Lead time",
                align: "right",
                render: (r) => `${r.lead_time_days} d`,
              },

              {
                key: "lifecycle",
                header: "Lifecycle",
                render: (r) => (
                  <Badge
                    tone={
                      r.lifecycle_stage === "maturity"
                        ? "success"
                        : "info"
                    }
                  >
                    {r.lifecycle_stage}
                  </Badge>
                ),
              },

              {
                key: "seasonality",
                header: "Seasonality",
                render: (r) => (
                  <Badge tone="info">
                    {r.seasonality_class}
                  </Badge>
                ),
              },

              {
                key: "volatility",
                header: "Volatility",
                render: (r) => (
                  <Badge tone="info">
                    {r.volatility_class}
                  </Badge>
                ),
              },
            ]}
          />
        )}

        {isFetching && !isLoading && (
          <div className="muted-note">Refreshing...</div>
        )}
      </Card>
    </>
  );
}