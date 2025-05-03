import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import { clearSignInNonce } from '@/utils/auth';

type Auth = {
  accountId: string;
};

type AuthStore = {
  auth: Auth | null;
  unauthorizedErrorHasTriggered: boolean;

  clearAuth: () => void;
  setAuth: (auth: Auth) => void;
  setUnauthorizedErrorHasTriggered: (value: boolean) => void;
};

// ✅ Define the store using Zustand's hook-based system
export const useNearAuthStore = create<AuthStore>()(
  devtools((set) => ({
    auth: null,
    unauthorizedErrorHasTriggered: false,

    clearAuth: () => {
      clearSignInNonce();
      set({
        auth: null,
        unauthorizedErrorHasTriggered: false,
      });
    },

    setAuth: (auth: Auth) => {
      set({ auth, unauthorizedErrorHasTriggered: false });
    },

    setUnauthorizedErrorHasTriggered: (value: boolean) => {
      set({ unauthorizedErrorHasTriggered: value });
    },
  }), { name: 'AuthStore' })
);

// ✅ Provide static accessors for outside React (NO `createStore`!)
export const nearAuthStore = {
  getState: useNearAuthStore.getState,
  setState: useNearAuthStore.setState,
};

export const name = 'AuthStore';
