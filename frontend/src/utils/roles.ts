import type { UserRole } from "@/state/authStore";

const ROLE_RANK: Record<UserRole, number> = {
  analyst: 1,
  reviewer: 2,
  procurement_manager: 3,
  supply_chain_manager: 4,
  admin: 5,
};

export function hasRole(userRole: UserRole | undefined, ...allowed: UserRole[]): boolean {
  if (!userRole) return false;
  if (allowed.length === 0) return true;
  return allowed.includes(userRole);
}

export function atLeast(userRole: UserRole | undefined, min: UserRole): boolean {
  if (!userRole) return false;
  return ROLE_RANK[userRole] >= ROLE_RANK[min];
}