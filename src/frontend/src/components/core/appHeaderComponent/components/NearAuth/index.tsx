import { useState, useEffect } from "react";
import NearIcon from "@/assets/near-icon.svg?react";
import { useWalletSelector } from "@near-wallet-selector/react-hook";

export function NearAuthIcon() {
    const { signedAccountId, signIn, signOut } = useWalletSelector();
    const [action, setAction] = useState<() => void>(() => {});
    const [label, setLabel] = useState("Loading...");

    useEffect(() => {
    
        if (signedAccountId) {
            setAction(() => signOut);
            setLabel(`Logout ${signedAccountId}`);
        } else {
            setAction(() => signIn);
            setLabel("Login");
        }
        }, [signedAccountId]);
  
  return (
      <div className="navbar-nav pt-1">
        <button
            className="flex items-center text-white px-4 py-2 rounded"
            onClick={() => action()}
        >
            <div className="relative">
                <NearIcon className="h-7 w-7 shrink-0 focus-visible:outline-0" />
                {signedAccountId && (
                    <svg
                    className="absolute top-0 left-5 h-5 w-5 text-green-500"
                    fill="currentColor"
                    viewBox="0 0 20 20"
                    xmlns="http://www.w3.org/2000/svg"
                    >
                    <path
                        fillRule="evenodd"
                        d="M16.293 5.293a1 1 0 0 0-1.414 0L8 11.586 5.121 8.707a1 1 0 0 0-1.414 1.414l3.535 3.535a1 1 0 0 0 1.414 0l8-8a1 1 0 0 0 0-1.414z"
                        clipRule="evenodd"
                    />
                    </svg>
                )}
            </div>
        </button>
      </div>
  );
}
