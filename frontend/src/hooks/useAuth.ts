import { useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { me } from "@/services/auth";
import { useAuthStore } from "@/state/authStore";

export function useAuth() {
  const { accessToken, user, setUser, clear } = useAuthStore();

  const { data, isLoading, isError } = useQuery({
    queryKey: ["me", accessToken],
    queryFn: me,
    enabled: !!accessToken,
    retry: false,
  });

  useEffect(() => {
    if (data) setUser(data);
    if (isError) clear();
  }, [data, isError, setUser, clear]);

  return {
    isAuthenticated: !!accessToken && !!user,
    user,
    isLoading,
  };
}