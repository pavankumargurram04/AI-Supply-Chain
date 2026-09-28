import { create } from "zustand";
import { persist } from "zustand/middleware";

export type UserRole =
  | "admin"
  | "analyst"
  | "procurement_manager"
  | "supply_chain_manager"
  | "reviewer";

export interface User {
  id: number;
  email: string;
  username: string;
  role: UserRole;
  full_name?: string | null;
  is_active: boolean;
}

interface AuthState {
  accessToken: string | null;
  refreshToken: string | null;
  user: User | null;
  setTokens: (a: string, r: string) => void;
  setUser: (u: User | null) => void;
  clear: () => void;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      accessToken: null,
      refreshToken: null,
      user: null,
      setTokens: (a, r) => set({ accessToken: a, refreshToken: r }),
      setUser: (u) => set({ user: u }),
      clear: () => set({ accessToken: null, refreshToken: null, user: null }),
    }),
    { name: "ai-sc-auth" },
  ),
);