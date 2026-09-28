import { api } from "./api";

export interface OrderListItem {
  id: number;
  po_number: string;
  supplier_id: number;
  warehouse_id: number;
  status: string;
  expected_delivery_date: string | null;
  actual_delivery_date: string | null;
  total_amount: number;
  currency: string;
  recommendation_id: number | null;
  created_at: string | null;
}

export interface OrderItem {
  id: number;
  product_id: number;
  quantity: number;
  received_quantity: number;
  unit_price: number;
  line_total: number;
}

export interface OrderShipment {
  id: number;
  carrier: string | null;
  tracking_number: string | null;
  status: string;
  expected_arrival: string | null;
  actual_arrival: string | null;
}

export interface OrderDetails extends OrderListItem {
  notes: string | null;
  items: OrderItem[];
  shipments: OrderShipment[];
}

export async function listOrders(
  status?: string,
): Promise<OrderListItem[]> {
  const r = await api.get<OrderListItem[]>("/api/v1/orders", {
    params: { status },
  });

  return r.data;
}

export async function getOrder(
  orderId: number,
): Promise<OrderDetails> {
  const r = await api.get<OrderDetails>(
    `/api/v1/orders/${orderId}`,
  );

  return r.data;
}