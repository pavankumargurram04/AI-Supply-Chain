import { api } from "./api";

export interface ReportEntry {
  name: string;
  path: string;
  size_bytes: number;
  modified_at: number;
}

export async function listReports(): Promise<ReportEntry[]> {
  const r = await api.get<ReportEntry[]>("/api/v1/reports");
  return r.data;
}

export async function getReport(name: string): Promise<any> {
  const r = await api.get(`/api/v1/reports/${name}`);
  return r.data;
}