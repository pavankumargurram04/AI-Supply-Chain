import { api } from "./api";

export interface Warehouse {
  id: number;
  code: string;
  name: string;
  region: string | null;
  country: string | null;
  capacity_units: number;
}

export async function listWarehouses(limit = 100): Promise<Warehouse[]> {
  const r = await api.get<Warehouse[]>("/api/v1/warehouses", { params: { limit } });
  return r.data;
}