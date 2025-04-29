const contractPerNetwork = {
    mainnet: "vitalpoint-donations.near"
  };
  
  export const NetworkId = "mainnet";
  export const nodeUrl = "https://rpc.mainnet.near.org";
  export const walletUrl = "https://app.mynearwallet.com";
  export const helperUrl = "https://helper.mainnet.near.org";
  export const DonationNearContract = contractPerNetwork[NetworkId];
  
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
  
  export const EVMWalletChain = evmWalletChains[NetworkId];