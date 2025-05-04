// src/components/core/RootLayout.tsx
import NearInitializer from "../../core/NearInitializer/NearInitializer";
import ZustandHydration from "../../core/NearInitializer/ZustandHydration";
import ContextWrapper from "@/contexts";
import FloatingDonationBox from "../FloatingDonationBox/FloatingDonationBox";
import { Outlet } from "react-router";

export default function RootLayout() {
  return (
    <ContextWrapper>
      <NearInitializer />
      <ZustandHydration />
      <Outlet />
      <FloatingDonationBox />
    </ContextWrapper>
  );
}
