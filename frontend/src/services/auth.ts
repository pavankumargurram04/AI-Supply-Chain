import { api } from "./api";
import type { User } from "@/state/authStore";

export interface TokenResponse {
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in: number;
}

export async function login(username: string, password: string): Promise<TokenResponse> {
  const r = await api.post<TokenResponse>("/api/v1/auth/login", { username, password });
  return r.data;
}

export async function me(): Promise<User> {
  const r = await api.get<User>("/api/v1/auth/me");
  return r.data;
}