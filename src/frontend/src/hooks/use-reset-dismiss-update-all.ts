import { useEffect } from "react";
import { useLocation } from "react-router";
import { useUtilityStore } from "../stores/utilityStore";
import { useShallow } from "zustand/react/shallow";

export const useResetDismissUpdateAll = () => {
  const location = useLocation();
  const flowLocationPath = location.pathname.includes("flow");
  const setDismissAll = useUtilityStore(useShallow((state) => state.setDismissAll));

  useEffect(() => {
    if (flowLocationPath) {
      setDismissAll(false);
    }
  }, [location]);
};
