import { setupMyNearWallet } from '@near-wallet-selector/my-near-wallet';
import { setupMeteorWallet } from '@near-wallet-selector/meteor-wallet';
import { setupMeteorWalletApp } from '@near-wallet-selector/meteor-wallet-app';
import { setupBitteWallet } from '@near-wallet-selector/bitte-wallet';
import { setupEthereumWallets } from '@near-wallet-selector/ethereum-wallets';
import { setupHotWallet } from '@near-wallet-selector/hot-wallet';
import { setupLedger } from '@near-wallet-selector/ledger';
import { setupSender } from '@near-wallet-selector/sender';
import { setupHereWallet } from '@near-wallet-selector/here-wallet';
import { setupNearMobileWallet } from '@near-wallet-selector/near-mobile-wallet';
import { setupWelldoneWallet } from '@near-wallet-selector/welldone-wallet';
import { WalletSelectorProvider } from '@near-wallet-selector/react-hook';
import { wagmiConfig, web3Modal } from '@/wallets/web3modal';
import { Navigation } from "../../components/donations/Navigation";
import { DonationNearContract } from "@/config";
import { WalletSelectorParams, WalletModuleFactory } from "@near-wallet-selector/core";
import "@/styles/globals.css";
import '@near-wallet-selector/modal-ui/styles.css';

interface ExtendedWalletSelectorParams extends WalletSelectorParams {
  createAccessKeyFor?: string;
}

type CustomWalletModuleFactory = WalletModuleFactory<any>;

const walletSelectorConfig: ExtendedWalletSelectorParams = {
  network: 'mainnet',
  createAccessKeyFor: DonationNearContract,
  modules: [
    setupEthereumWallets({ wagmiConfig, web3Modal, alwaysOnboardDuringSignIn: true }) as CustomWalletModuleFactory,
    setupBitteWallet() as CustomWalletModuleFactory,
    setupMeteorWallet(),
    setupMeteorWalletApp({contractId: DonationNearContract}) as CustomWalletModuleFactory,
    setupHotWallet(),
    setupLedger(),
    setupSender(),
    setupHereWallet(),
    setupNearMobileWallet(),
    setupWelldoneWallet(),
    setupMyNearWallet(),
  ],
}

export default function WalletSetup({ Component, pageProps }) {

  return (
    <WalletSelectorProvider config={walletSelectorConfig}>
      <Navigation />
      <Component {...pageProps} />
    </WalletSelectorProvider>
  );
}