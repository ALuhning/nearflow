// src/utils/nearConnection.ts

import { connect, keyStores, utils, providers, WalletConnection } from "near-api-js";
import { create } from "zustand";

interface NearStore {
  near: any | null; // The NEAR connection object
  accountId: string | null; // The current account ID
  wallet: any | null; // The current wallet object
  setNear: (near: any) => void; // Method to set the NEAR connection
  setAccountId: (accountId: string | null) => void; // Method to set the account ID
  setWallet: (wallet: any) => void; // Method to set the wallet object
}

export const useNearStore = create<NearStore>((set) => ({
  near: null,
  accountId: null,
  wallet: null,
  setNear: (near) => set({ near }),
  setAccountId: (accountId) => set({ accountId }),
  setWallet: (wallet) => set({ wallet }),
}));

const setupNearConnection = async (config: any) => {
  const { networkId, nodeUrl } = config;

  const near = await connect({
    networkId,
    nodeUrl,
    keyStore: new keyStores.BrowserLocalStorageKeyStore(), // Store the key in localStorage
  });

  // Store the NEAR connection and wallet in Zustand store
  useNearStore.getState().setNear(near);

  const wallet = new WalletConnection(near, 'nearflow'); // Get wallet object
  useNearStore.getState().setWallet(wallet);

  return { near, wallet };
};

export const getAccountId = async () => {
  const { wallet } = useNearStore.getState();
  if (wallet) {
    const accountId = wallet.accountId;
    useNearStore.getState().setAccountId(accountId);
    return accountId;
  }
  return null;
};

export const viewFunction = async (contractId: string, method: string, args: object) => {
  const { near } = useNearStore.getState();
  if (!near) throw new Error("NEAR connection is not established");

  const provider = new providers.JsonRpcProvider(near.config.nodeUrl);

  const res = await provider.query({
    request_type: "call_function",
    account_id: contractId,
    method_name: method,
    args_base64: Buffer.from(JSON.stringify(args)).toString("base64"),
    finality: "optimistic",
  });

  // Narrow the type to ensure TypeScript knows `result` exists
  const callFunctionResult = res as unknown as { result: Uint8Array };

  if (!callFunctionResult.result) {
    throw new Error("Empty result from contract");
  }

  return JSON.parse(Buffer.from(callFunctionResult.result).toString());
};

export const callFunction = async (contractId: string, method: string, args: object, deposit: string) => {
  const { wallet } = useNearStore.getState();
  if (!wallet) throw new Error("Wallet is not connected");

  const outcome = await wallet.signAndSendTransaction({
    receiverId: contractId,
    actions: [
      {
        type: "FunctionCall",
        params: {
          methodName: method,
          args,
          gas: "30000000000000", // You can customize gas
          deposit,
        },
      },
    ],
  });

  return providers.getTransactionLastResult(outcome);
};

export default setupNearConnection;
