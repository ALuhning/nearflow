// authStore.js
import { LANGFLOW_ACCESS_TOKEN, LANGFLOW_REFRESH_TOKEN, LANGFLOW_API_TOKEN, LANGFLOW_AUTO_LOGIN_OPTION } from "@/constants/constants";
import { AuthStoreType } from "@/types/zustand/auth";
import { Cookies } from "react-cookie";
import { create } from "zustand";

const cookies = new Cookies();
const useAuthStore = create<AuthStoreType>((set, get) => ({
  isAdmin: false,
  isAuthenticated: !!cookies.get(LANGFLOW_ACCESS_TOKEN),
  accessToken: cookies.get(LANGFLOW_ACCESS_TOKEN) ?? null,
  userData: null,
  autoLogin: null,
  apiKey: cookies.get(LANGFLOW_API_TOKEN),
  authenticationErrorCount: 0,

  setIsAdmin: (isAdmin) => {
    console.log(`AuthStore: Manually setting isAdmin to ${isAdmin}`);
    set({ isAdmin });
  },
  setIsAuthenticated: (isAuthenticated) => set({ isAuthenticated }),
  setAccessToken: (accessToken) => set({ accessToken }),
  setUserData: (userData) => {
    console.log("AuthStore: Setting user data:", userData);
    set({ userData });
    // Automatically update isAdmin based on user data
    if (userData && typeof userData.is_superuser === 'boolean') {
      console.log(`AuthStore: Setting isAdmin to ${userData.is_superuser} for user ${userData.username || userData.id}`);
      set({ isAdmin: userData.is_superuser });
    } else if (!userData) {
      // Clear isAdmin when userData is cleared
      console.log("AuthStore: Clearing userData and isAdmin");
      set({ isAdmin: false });
    }
  },
  setAutoLogin: (autoLogin) => set({ autoLogin }),
  setApiKey: (apiKey) => set({ apiKey }),
  setAuthenticationErrorCount: (authenticationErrorCount) =>
    set({ authenticationErrorCount }),

  logout: async () => {
    console.log("AuthStore: Logging out - clearing all auth state");
    // Clear cookies
    cookies.remove(LANGFLOW_ACCESS_TOKEN, { path: "/" });
    cookies.remove(LANGFLOW_REFRESH_TOKEN, { path: "/" });
    cookies.remove(LANGFLOW_API_TOKEN, { path: "/" });
    cookies.remove(LANGFLOW_AUTO_LOGIN_OPTION, { path: "/" });

    get().setIsAuthenticated(false);
    get().setIsAdmin(false);

    set({
      isAdmin: false,
      userData: null,
      accessToken: null,
      isAuthenticated: false,
      autoLogin: false,
      apiKey: null,
    });
    console.log("AuthStore: Logout completed");
  },
}));

export default useAuthStore;
