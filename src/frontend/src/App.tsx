import "@xyflow/react/dist/style.css";
import { Suspense, useEffect } from "react";
import { RouterProvider } from "react-router-dom";
import { LoadingPage } from "./pages/LoadingPage";
import router from "./routes";
import { useDarkStore } from "./stores/darkStore";

import { setupMyNearWallet } from '@near-wallet-selector/my-near-wallet';
import { setupMeteorWallet } from '@near-wallet-selector/meteor-wallet';
import { setupMeteorWalletApp } from '@near-wallet-selector/meteor-wallet-app';
import { setupBitteWallet } from '@near-wallet-selector/bitte-wallet';
import { setupHotWallet } from '@near-wallet-selector/hot-wallet';
import { setupSender } from '@near-wallet-selector/sender';
import { setupHereWallet } from '@near-wallet-selector/here-wallet';
import { setupNearMobileWallet } from '@near-wallet-selector/near-mobile-wallet';
import { setupWelldoneWallet } from '@near-wallet-selector/welldone-wallet';
import { DonationNearContract } from "@/config";
import { WalletSelectorProvider } from "@near-wallet-selector/react-hook";
import { WalletModuleFactory } from "@near-wallet-selector/core";
import '@near-wallet-selector/modal-ui/styles.css';

type CustomWalletModuleFactory = WalletModuleFactory<any>;

const walletSelectorConfig = {
  network: 'mainnet',
  createAccessKeyFor: DonationNearContract,
  modules: [
    setupBitteWallet() as CustomWalletModuleFactory,
    setupMeteorWallet(),
    setupMeteorWalletApp({contractId: DonationNearContract}) as CustomWalletModuleFactory,
    setupHotWallet(),
    setupSender(),
    setupHereWallet(),
    setupNearMobileWallet(),
    setupWelldoneWallet(),
    setupMyNearWallet(),
  ],
};

export default function App() {
  const dark = useDarkStore((state) => state.dark);

  // Initialize wallet selector on component mount
  useEffect(() => {
    if (!dark) {
      document.getElementById("body")!.classList.remove("dark");
    } else {
      document.getElementById("body")!.classList.add("dark");
    }
  }, [dark]);

  return (
    <>
      <Suspense fallback={<LoadingPage />}>
        <WalletSelectorProvider config={walletSelectorConfig}>
          <RouterProvider router={router}/>
        </WalletSelectorProvider>
      </Suspense>
    </>
  );
}
