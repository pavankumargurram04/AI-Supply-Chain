import { api } from "./api";

export interface Forecast {
  product_sku: string;
  warehouse_code: string;
  horizon: number;
  origin_date: string;
  forecast_dates: string[];
  point: number[];
  lower: number[] | null;
  upper: number[] | null;
  quantiles: Record<string, number[]> | null;
  model_version: string;
  feature_version: string;
  seed: number;
}

export async function getForecast(
  product_sku: string,
  warehouse_code: string,
  mode = "dev",
): Promise<Forecast> {
  const r = await api.get<Forecast>(
    `/api/v1/forecast/${product_sku}/${warehouse_code}`,
    { params: { mode } },
  );
  return r.data;
}

export async function predict(
  product_sku: string,
  warehouse_code: string,
  mode = "dev",
  origin_date?: string,
  mc_dropout_samples = 0,
): Promise<Forecast> {
  const r = await api.post<Forecast>(
    "/api/v1/forecast/predict",
    { product_sku, warehouse_code, mode, origin_date, mc_dropout_samples },
  );
  return r.data;
}