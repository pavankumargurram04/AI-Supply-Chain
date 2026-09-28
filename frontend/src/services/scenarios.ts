import { api } from "./api";

export interface ScenarioInput {
  product_sku: string;
  warehouse_code: string;
  mode?: string;
  scenario_type?: string;
  demand_multiplier?: number;
  lead_time_override_days?: number | null;
  inventory_override?: number | null;
  budget?: number | null;
  seed?: number;
}

export interface ScenarioDelta {
  final_quantity: number;
  total_cost: number;
  stockout_probability: number;
  safety_stock: number;
  reorder_point: number;
  expected_service_level: number;
}

export interface ScenarioResult {
  scenario_id: number;
  baseline: {
    final_quantity: number;
    total_cost: number;
    stockout_probability: number;
    safety_stock: number;
    reorder_point: number;
    expected_service_level: number;
  };
  scenario: ScenarioResult["baseline"];
  delta: ScenarioDelta;
}

export async function runScenario(input: ScenarioInput): Promise<ScenarioResult> {
  const r = await api.post<ScenarioResult>("/api/v1/scenarios/run", input);
  return r.data;
}

export async function getScenario(id: number): Promise<any> {
  const r = await api.get(`/api/v1/scenarios/${id}`);
  return r.data;
}