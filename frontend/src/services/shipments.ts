import { api } from "./api";

export interface Shipment {
  id: number;
  purchase_order_id: number;
  po_number: string;
  carrier: string | null;
  tracking_number: string | null;
  status: string;
  dispatched_at: string | null;
  expected_arrival: string | null;
  actual_arrival: string | null;
}

export async function listShipments(status?: string): Promise<Shipment[]> {
  const r = await api.get<Shipment[]>("/api/v1/shipments", { params: { status } });
  return r.data;
}

export async function getShipment(id: number): Promise<any> {
  const r = await api.get(`/api/v1/shipments/${id}`);
  return r.data;
}