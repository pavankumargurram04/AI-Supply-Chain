import { api } from "./api";

export interface ModelRow {
  id: number;
  model_name: string;
  family: string;
  version: string;
  status: string;
  dataset_version: string | null;
  feature_version: string | null;
  metrics: Record<string, unknown> | null;
  trained_at: string | null;
  promoted_at: string | null;
}

export async function listModels(family?: string, status?: string): Promise<ModelRow[]> {
  const r = await api.get<ModelRow[]>("/api/v1/models", { params: { family, status } });
  return r.data;
}

export async function getModel(model_name: string, version: string): Promise<any> {
  const r = await api.get(`/api/v1/models/${model_name}/${version}`);
  return r.data;
}

export async function promote(
  model_name: string,
  version: string,
  thresholds: any[] = [],
  primary = "test.wape",
): Promise<any> {
  const r = await api.post(`/api/v1/models/${model_name}/${version}/promote`, {
    thresholds,
    primary_metric_path: primary,
    primary_metric_direction: "min",
    min_relative_improvement: 0,
  });
  return r.data;
}

export async function rollback(model_name: string, reason: string): Promise<any> {
  const r = await api.post(`/api/v1/models/${model_name}/rollback`, null, {
    params: { reason },
  });
  return r.data;
}