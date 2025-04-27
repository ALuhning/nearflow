import { useEffect, useState } from "react";
import NearLogo from "@/assets/near-logo.svg";
//import { useWalletSelector } from '@/lib/useWalletSelector';
import { useWalletSelector } from "@near-wallet-selector/react-hook";

export const Navigation = () => {
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
    <nav className="navbar navbar-expand-lg">
      <div className="container-fluid">
          <img
            src={NearLogo}
            alt="NEAR"
            width="30"
            height="24"
            className="d-inline-block align-text-top"
          />
        <div className="navbar-nav pt-1">
          <button className="btn btn-secondary" onClick={() => action()}>
            {" "}
            {label}{" "}
          </button>
        </div>
      </div>
    </nav>
  );
};