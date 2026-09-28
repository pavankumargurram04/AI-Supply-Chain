import { api } from "./api";

export interface Supplier {
  id: number;
  code: string;
  name: string;
  country: string | null;
  region: string | null;
  reliability_class: string;
}

export async function listSuppliers(reliability?: string, limit = 100): Promise<Supplier[]> {
  const r = await api.get<Supplier[]>("/api/v1/suppliers", { params: { reliability, limit } });
  return r.data;
}

export async function getSupplier(id: number): Promise<Supplier> {
  const r = await api.get<Supplier>(`/api/v1/suppliers/${id}`);
  return r.data;
}