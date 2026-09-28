import { api } from "./api";

export interface SupplierRisk {
  supplier_code: string;
  risk_probability: number;
  risk_label: string;
  lead_time_days: number;
  unit_price: number;
  moq: number;
  is_preferred: boolean;
  model_version: string | null;
}

export async function getSupplierRisk(
  product_sku: string,
  mode = "dev",
): Promise<SupplierRisk[]> {
  const r = await api.get<SupplierRisk[]>("/api/v1/supplier-risk", {
    params: { product_sku, mode },
  });
  return r.data;
}