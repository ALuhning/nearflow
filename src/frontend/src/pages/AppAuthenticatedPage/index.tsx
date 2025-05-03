import { useCustomPostAuth } from "@/customization/hooks/use-custom-post-auth";
import { Outlet } from "react-router";

export function AppAuthenticatedPage() {
  useCustomPostAuth();

  return <Outlet />;
}
