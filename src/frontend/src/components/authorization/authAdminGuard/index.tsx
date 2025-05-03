import { AuthContext } from "@/contexts/authContext";
import { CustomNavigate } from "@/customization/components/custom-navigate";
import { LoadingPage } from "@/pages/LoadingPage";
import useAuthStore from "@/stores/authStore";
import { useShallow } from "zustand/react/shallow";
import { useContext } from "react";

export const ProtectedAdminRoute = ({ children }) => {
  const { userData } = useContext(AuthContext);
  const isAuthenticated = useAuthStore(useShallow((state) => state.isAuthenticated));
  const autoLogin = useAuthStore(useShallow((state) => state.autoLogin));
  const isAdmin = useAuthStore(useShallow((state) => state.isAdmin));

  if (!isAuthenticated) {
    return <LoadingPage />;
  } else if ((userData && !isAdmin) || autoLogin) {
    return <CustomNavigate to="/" replace />;
  } else {
    return children;
  }
};
