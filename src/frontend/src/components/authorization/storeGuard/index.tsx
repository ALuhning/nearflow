import { CustomNavigate } from "@/customization/components/custom-navigate";
import { useStoreStore } from "../../../stores/storeStore";
import { useShallow } from "zustand/react/shallow";

export const StoreGuard = ({ children }) => {
  const hasStore = useStoreStore(useShallow((state) => state.hasStore));

  if (!hasStore) {
    return <CustomNavigate to="/all" replace />;
  }

  return children;
};
