import StoreApiKeyPage from "@/pages/SettingsPage/pages/StoreApiKeyPage";
import { Route } from "react-router";

export const CustomRoutesStore = () => {
  return (
    <>
      <Route path="store" element={<StoreApiKeyPage />} />
    </>
  );
};

export default CustomRoutesStore;
