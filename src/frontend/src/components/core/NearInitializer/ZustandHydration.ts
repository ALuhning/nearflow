import { useEffect } from 'react';

import { useNearAuthStore } from '@/stores/nearAuthStore';
import { useSessionQuery } from "@/controllers/API/queries/auth/use-post-login-user-near";
import { useShallow } from 'zustand/react/shallow';

export const ZustandHydration = () => {
  const { data: session } = useSessionQuery();
  const { setAuth, clearAuth } = useNearAuthStore();
  const unauthorizedErrorHasTriggered = useNearAuthStore(useShallow((store) => store.unauthorizedErrorHasTriggered));
  
  useEffect(() => {
    if (session && !unauthorizedErrorHasTriggered) {
      setAuth(session);
    }
    if (unauthorizedErrorHasTriggered) {
      clearAuth();
    }
  }, [session, unauthorizedErrorHasTriggered]);

  return null;
};

export default ZustandHydration;