import { useState, useEffect } from "react";
import NearIcon from "@/assets/near-icon.svg?react";
import { WalletSelectorModal } from "@near-wallet-selector/modal-ui";
import { useWalletStore } from '@/stores/walletStore';

export function NearAuthIcon() {
  const [modal, setModal] = useState<WalletSelectorModal | null>(null);
  const wallet = useWalletStore((store) => store.wallet);
  const walletModal = useWalletStore((store) => store.modal);
  const walletAccount = useWalletStore((store) => store.account);

  console.log("walletmodal", walletModal);
  return (
    <div>
      <NearIcon
        className="h-7 w-7 shrink-0 focus-visible:outline-0"
        onClick={() => walletModal?.show()}
      />
    </div>
  );
}
