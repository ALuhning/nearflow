export const config = {
  AUTH_NEAR_URL: "https://auth.near.ai",
  AUTH_COOKIE_DOMAIN: "https://near.ai",
  RECIPIENT: "ai.near",
  MESSAGE: "Welcome to NEAR AI Hub!",
  REVOKE_MESSAGE: "Are you sure? Revoking a nonce",
  REVOKE_ALL_MESSAGE: "Are you sure? Revoking all nonces",
  SIGN_IN_CALLBACK_PATH: "/sign-in/callback",
  SIGN_IN_RESTORE_URL_KEY: "signInRestoreUrl",
  SIGN_IN_NONCE_KEY: "signInNonce",
  NODE_ENV: "development",
  PUBLIC_BASE_URL: "https://app.near.ai",
  network: {
    networkId: 'mainnet',
    nodeUrl: 'https://rpc.mainnet.fastnear.com',
    walletUrl: 'https://app.mynearwallet.com',
    helperUrl: 'https://helper.mainnet.near.org',
    explorerUrl: 'https://explorer.mainnet.near.org',
    indexerUrl: 'https://indexer.mainnet.near.org',
  },
  contractPerNetwork: {
    mainnet: "vitalpoint-donations.near"
  }
}

// Chains for EVM Wallets
const evmWalletChains = {
  mainnet: {
    chainId: 397,
    name: "Near Mainnet",
    explorer: "https://eth-explorer.near.org",
    rpc: "https://eth-rpc.mainnet.near.org",
  },
  testnet: {
    chainId: 398,
    name: "Near Testnet",
    explorer: "https://eth-explorer-testnet.near.org",
    rpc: "https://eth-rpc.testnet.near.org",
  },
};

export const DonationNearContract = config.contractPerNetwork[config.network.networkId];
export const EVMWalletChain = evmWalletChains[config.network.networkId];