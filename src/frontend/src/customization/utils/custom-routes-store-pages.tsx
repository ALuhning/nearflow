import { StoreGuard } from "@/components/authorization/storeGuard";
import StorePage from "@/pages/StorePage";
import { Route } from "react-router";

export const CustomRoutesStorePages = () => {
  return (
    <>
      <Route
        path="store"
        element={
          <StoreGuard>
            <StorePage />
          </StoreGuard>
        }
      />
      <Route
        path="store/:id/"
        element={
          <StoreGuard>
            <StorePage />
          </StoreGuard>
        }
      />
    </>
  );
};

export default CustomRoutesStorePages;
