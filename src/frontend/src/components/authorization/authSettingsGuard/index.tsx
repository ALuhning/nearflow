import { CustomNavigate } from "@/customization/components/custom-navigate";
import { ENABLE_PROFILE_ICONS } from "@/customization/feature-flags";
import useAuthStore from "@/stores/authStore";
import { useStoreStore } from "@/stores/storeStore";
import { useShallow } from "zustand/react/shallow";

export const AuthSettingsGuard = ({ children }) => {
  const autoLogin = useAuthStore(useShallow((state) => state.autoLogin));
  const hasStore = useStoreStore(useShallow((state) => state.hasStore));

  // Hides the General settings if there is nothing to show
  const showGeneralSettings = ENABLE_PROFILE_ICONS || hasStore || !autoLogin;

  if (showGeneralSettings) {
    return children;
  } else {
    return <CustomNavigate replace to="/settings/global-variables" />;
  }
};
