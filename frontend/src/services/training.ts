import { api } from "./api";

export interface TrainingJob {
  id: number;
  model_family: string;
  status: string;
  seed: number;
  config: Record<string, unknown> | null;
  metrics: Record<string, unknown> | null;
  started_at: string | null;
  finished_at: string | null;
  error_message: string | null;
}

export async function enqueueTraining(
  model_family: string,
  mode: string,
  seed = 42,
): Promise<{ job_id: number; status: string; dispatched: boolean }> {
  const r = await api.post("/api/v1/training/jobs", { model_family, mode, seed });
  return r.data;
}

export async function listJobs(status?: string): Promise<any[]> {
  const r = await api.get<any[]>("/api/v1/training/jobs", { params: { status } });
  return r.data;
}

export async function getJob(id: number): Promise<TrainingJob> {
  const r = await api.get<TrainingJob>(`/api/v1/training/jobs/${id}`);
  return r.data;
}