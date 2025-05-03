import { useCallback } from "react";
import { Buffer } from "buffer";
import { useWalletStore } from "@/stores/walletStore";
import { useSignAuthUrlMessageMutation } from "@/controllers/API/queries/auth/use-post-login-user-near";

export function useSignAuthUrlMessage() {
  const selector = useWalletStore((state) => state.selector);
  const { mutate: signMessage } = useSignAuthUrlMessageMutation();

  const signFromAuthUrl = useCallback(
    async (authUrl: string) => {
      if (!selector) return;
      
      const parsed = new URL(authUrl);
      const message = parsed.searchParams.get("message");
      const recipient = parsed.searchParams.get("recipient");
      const nonce = parsed.searchParams.get("nonce");
      const callbackUrl = parsed.searchParams.get("callbackUrl") || parsed.origin;

      if (!message || !recipient || !nonce) {
        throw new Error("Missing required URL params");
      }

      const wallet = await selector.wallet();
      const account = selector.store.getState().accounts[0];

      if (!wallet || !account) throw new Error("No wallet or account selected");

      const signed = await wallet.signMessage({
        message: message,
        recipient,
        nonce: Buffer.from(nonce, "utf-8"),
      });
     
      if (!signed) throw new Error("Signature failed")

      const res = signMessage({
        accountId: account.accountId,
        publicKey: signed.publicKey,
        signature: signed.signature,
        message,
        nonce,
        recipient,
        callbackUrl
      });
      return res;
    },
    [selector]
  );

  return { signFromAuthUrl };
}
