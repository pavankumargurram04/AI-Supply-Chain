import { api } from "./api";

export interface RecommendationListItem {
  id: number;
  trace_id: string;
  product_id: number;
  warehouse_id: number;
  status: string;
  final_quantity: number | null;
  expected_cost: number | null;
  expected_service_level: number | null;
  stockout_probability: number | null;
  created_at: string;
}

export async function listRecommendations(status?: string): Promise<RecommendationListItem[]> {
  const r = await api.get<RecommendationListItem[]>("/api/v1/procurement", {
    params: { status },
  });
  return r.data;
}

export async function approve(id: number, notes?: string): Promise<RecommendationListItem> {
  const r = await api.post(`/api/v1/procurement/${id}/approve`, { notes });
  return r.data;
}

export async function reject(id: number, notes?: string): Promise<RecommendationListItem> {
  const r = await api.post(`/api/v1/procurement/${id}/reject`, { notes });
  return r.data;
}

export async function override(
  id: number,
  new_quantity: number,
  reason: string,
): Promise<RecommendationListItem> {
  const r = await api.post(`/api/v1/procurement/${id}/override`, { new_quantity, reason });
  return r.data;
}

export async function createOrderFromRecommendation(id: number): Promise<{
  purchase_order_id: number;
  po_number: string;
  shipment_id: number;
  total_amount: number;
  expected_delivery_date: string;
  n_items: number;
}> {
  const r = await api.post(`/api/v1/procurement/${id}/create-order`);
  return r.data;
}

export interface CreateRecommendationInput {
  product_sku: string;
  warehouse_code: string;
  service_level?: number;
  horizon_days?: number;
  seed?: number;
}

export async function createRecommendation(
  input: CreateRecommendationInput,
): Promise<RecommendationListItem> {
  const r = await api.post<RecommendationListItem>("/api/v1/procurement/recommend", {
    service_level: 0.95,
    horizon_days: 30,
    seed: 42,
    ...input,
  });
  return r.data;
}