import { api } from "./api";

export interface Product {
  id: number;
  sku: string;
  name: string;
  description: string | null;
  category_id: number | null;

  unit_cost: number;
  unit_price: number;
  weight_kg: number | null;
  volume_m3: number | null;

  lifecycle_stage: string;
  seasonality_class: string;
  volatility_class: string;

  base_demand: number;
  lead_time_days: number;
}

export async function listProducts(
  search?: string,
  limit = 50
): Promise<Product[]> {
  const r = await api.get<Product[]>("/api/v1/products", {
    params: { search, limit },
  });

  return r.data;
}

export async function getProduct(id: number): Promise<Product> {
  const r = await api.get<Product>(`/api/v1/products/${id}`);
  return r.data;
}