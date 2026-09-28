import { api } from "./api";

export interface Alert {
  id: number;
  severity: string;
  status: string;
  source: string;
  title: string;
  message: string;
  entity_type: string | null;
  entity_id: number | null;
  created_at: string;
  payload: Record<string, unknown> | null;
}

export interface UsageSnapshot {
  window_days: number;
  recommendations: { total: number; by_status: Record<string, number> };
  predictions: { total: number };
  audit_events: { total: number };
  approval_funnel: {
    generated: number;
    reviewed: number;
    approved_or_overridden: number;
    rejected: number;
    order_created: number;
  };
}

export async function listAlerts(status?: string, severity?: string): Promise<Alert[]> {
  const r = await api.get<Alert[]>("/api/v1/monitoring/alerts", {
    params: { status, severity },
  });
  return r.data;
}

export async function getUsage(days = 7): Promise<UsageSnapshot> {
  const r = await api.get<UsageSnapshot>("/api/v1/monitoring/usage", { params: { days } });
  return r.data;
}