import { api } from "./api";

export interface AgentMessage {
  message_id: string;
  trace_id: string;
  agent: string;
  timestamp: string;
  status: string;
  confidence: number | null;
  payload: Record<string, unknown>;
  warnings: string[];
  errors: string[];
  model_versions: Record<string, string>;
}

export interface AgentRun {
  trace_id: string;
  run_at: string;
  messages: AgentMessage[];
  decision_trace: Record<string, unknown>;
}

export interface AgentRunInput {
  product_sku: string;
  warehouse_code: string;
  service_level?: number;
  horizon_days?: number;
  demand_multiplier?: number;
  seed?: number;
}

export async function runAgents(input: AgentRunInput): Promise<AgentRun> {
  const r = await api.post<AgentRun>("/api/v1/agents/run", input);
  return r.data;
}

export async function getAgentTrace(trace_id: string): Promise<any> {
  const r = await api.get(`/api/v1/agents/trace/${trace_id}`);
  return r.data;
}