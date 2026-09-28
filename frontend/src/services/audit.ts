import { api } from "./api";

export interface AuditEntry {
  id: number;
  actor_id: number | null;
  actor_name: string | null;
  action: string;
  entity_type: string;
  entity_id: number | null;
  before: Record<string, unknown> | null;
  after: Record<string, unknown> | null;
  request_id: string | null;
  trace_id: string | null;
  notes: string | null;
  created_at: string;
}

export async function listAudit(filters?: {
  entity_type?: string;
  action?: string;
  trace_id?: string;
  limit?: number;
}): Promise<AuditEntry[]> {
  const r = await api.get<AuditEntry[]>("/api/v1/audit", { params: filters });
  return r.data;
}