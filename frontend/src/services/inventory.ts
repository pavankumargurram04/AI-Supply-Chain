import { api } from "./api";

export interface InventoryState {
  product_sku: string;
  warehouse_code: string;
  on_hand: number;
  reserved: number;
  inbound: number;
  available: number;
  inventory_position: number;
}

export interface DecisionTrace {
  trace_id: string;
  run_at: string;
  product_sku: string;
  warehouse_code: string;
  final_quantity: number;
  total_cost: number;
  expected_service_level: number;
  stockout_probability: number;
  safety_stock: number;
  reorder_point: number;
  confidence: number;
  guardrail_violations: string[];
  explanation: string;
  full_trace: Record<string, unknown>;
}

export interface RecommendInput {
  product_sku: string;
  warehouse_code: string;
  service_level?: number;
  horizon_days?: number;
  demand_multiplier?: number;
  lead_time_override_days?: number | null;
  inventory_override?: number | null;
  budget?: number | null;
  max_suppliers?: number | null;
  seed?: number;
}

export async function getInventoryState(
  product_sku: string,
  warehouse_code: string,
  mode = "dev",
): Promise<InventoryState> {
  const r = await api.get<InventoryState>(
    `/api/v1/inventory/${product_sku}/${warehouse_code}`,
    { params: { mode } },
  );
  return r.data;
}

export async function recommend(input: RecommendInput, mode = "dev"): Promise<DecisionTrace> {
  const r = await api.post<DecisionTrace>("/api/v1/inventory/recommend", input, {
    params: { mode },
  });
  return r.data;
}