import { useEffect, useState } from "react";
import { useNavigate } from "react-router";
import NearIcon from "@/assets/near-icon.svg";
import {
  returnUrlToRestoreAfterSignIn,
  generateNonce,
  createAuthUrl,
} from "@/utils/auth";
import { config } from "@/config";
import { useNearAuthStore } from "@/stores/nearAuthStore";
import { useWalletStore } from "@/stores/walletStore";
import { useShallow } from "zustand/react/shallow";
import { useSignAuthUrlMessage } from "./SignIn";
import { usePostGlobalVariables } from "@/controllers/API/queries/variables/use-post-global-variables";
import { usePatchGlobalVariables } from "@/controllers/API/queries/variables/use-patch-global-variables";
import { useGetGlobalVariables } from "@/controllers/API/queries/variables";
import Cookies from "js-cookie";

export default function NearAuthIcon() {
  const { MESSAGE, RECIPIENT } = config;
  const navigate = useNavigate();
  const selector = useWalletStore((state) => state.selector);
  const walletAccount = useWalletStore(useShallow((s) => s.account));
  const wallet = useWalletStore(useShallow((state) => state.wallet));
  const walletModal = useWalletStore(useShallow((state) => state.modal));
  const { auth, setAuth, clearAuth } = useNearAuthStore();

  const { signFromAuthUrl } = useSignAuthUrlMessage();
  const { data: variables } = useGetGlobalVariables();
  const { mutateAsync: postGlobalVar } = usePostGlobalVariables();
  const { mutateAsync: patchGlobalVar } = usePatchGlobalVariables();

  const [walletConnected, setWalletConnected] = useState(false);
  const [readyToSign, setReadyToSign] = useState(false);
  const [toggle, setToggle] = useState(false);
  const [hasNearAuth, setHasNearAuth] = useState(false);
  const [hovered, setHovered] = useState<"top" | "bottom" | null>(null);

  const authUrl = createAuthUrl(MESSAGE, RECIPIENT, generateNonce());

  useEffect(() => {
    const cookieMatch = document.cookie
      .split("; ")
      .find((row) => row.startsWith("auth="));
    if (!cookieMatch) return;

    try {
      const encodedValue = decodeURIComponent(cookieMatch.split("=")[1]);
      const decodedJson = atob(encodedValue);
      const cookieObject = JSON.parse(decodedJson);
      if (cookieObject?.account_id) {
        setAuth({ accountId: cookieObject.account_id });
        setHasNearAuth(true);
      }
    } catch (err) {
      console.warn("Failed to restore auth from cookie:", err);
    }
  }, []);

  useEffect(() => {
    if (!selector) return;
    const unsubscribe = selector.store.observable.subscribe(({ accounts }) => {
      setWalletConnected(accounts.length > 0);
    });
    return () => unsubscribe.unsubscribe();
  }, [selector]);

  useEffect(() => {
    if (toggle) walletModal?.show();
    if (walletConnected) walletModal?.hide();
  }, [toggle, walletModal]);

  useEffect(() => {
    if (!readyToSign || !walletConnected) return;
    (async () => {
      try {
        const response = await signFromAuthUrl(authUrl);
        const accountId = walletAccount?.accountId ?? "";
        setAuth({ accountId });
        setHasNearAuth(true);
        navigate(returnUrlToRestoreAfterSignIn());

        const cookieMatch = document.cookie
          .split("; ")
          .find((row) => row.startsWith("auth="));
        if (!cookieMatch) return;
        const encodedValue = decodeURIComponent(cookieMatch.split("=")[1]);
        const decodedJson = atob(encodedValue);
        const cookieObject = JSON.parse(decodedJson);
        const upsertValue = JSON.stringify({ auth: cookieObject });
        const existing = variables?.find((v) => v.name === "NEARAI");

        if (existing) {
          await patchGlobalVar({
            id: existing.id,
            name: "NEARAI",
            value: upsertValue,
            default_fields: existing.default_fields ?? ["Near Credentials"],
          });
        } else {
          await postGlobalVar({
            name: "NEARAI",
            value: upsertValue,
            type: "Credential",
            default_fields: ["Near Credentials"],
          });
        }
      } catch (err) {
        console.error("Sign message or upsert failed:", err);
      } finally {
        setReadyToSign(false);
      }
    })();
  }, [readyToSign]);

  useEffect(() => {
    if (
      !toggle ||
      !walletAccount ||
      !auth?.accountId ||
      walletAccount.accountId !== auth.accountId
    )
      return;
    (async () => {
      try {
        await wallet?.signOut();
        clearAuth();
        setWalletConnected(false);
        setHasNearAuth(false);
        navigate(returnUrlToRestoreAfterSignIn());
      } catch (err) {
        console.error("Sign out failed:", err);
      } finally {
        setToggle(false);
      }
    })();
  }, [toggle, walletAccount, auth]);

  const handleClick = () => setToggle((t) => !t);

  const showSignButton = walletConnected && !auth?.accountId;

  const Icon = ({
    status,
    type,
    onClick,
    pending = false,
    setHovered,
  }: {
    status: boolean;
    type: "top" | "bottom";
    onClick: () => void;
    pending?: boolean;
    setHovered: (val: "top" | "bottom" | null) => void;
  }) => (
    <div
      className={`absolute cursor-pointer ${
        type === "top" ? "top-0" : "bottom-0"
      } -right-2`}
      onClick={(e) => {
        e.stopPropagation();
        onClick();
      }}
      onMouseEnter={() => setHovered(type)}
      onMouseLeave={() => setHovered(null)}
    >
      <svg
        className={`h-3.5 w-3.5 ${
          pending
            ? "text-yellow-400 animate-pulse"
            : status
            ? "text-green-500"
            : "text-red-500"
        }`}
        fill="currentColor"
        viewBox="0 0 20 20"
      >
        <path
          fillRule="evenodd"
          d={
            pending
              ? "M10 2a8 8 0 100 16 8 8 0 000-16zm1 4H9v5h4v-2h-3V6z" // clock icon
              : status
              ? "M16.293 5.293a1 1 0 0 0-1.414 0L8 11.586 5.121 8.707a1 1 0 0 0-1.414 1.414l3.535 3.535a1 1 0 0 0 1.414 0l8-8a1 1 0 0 0 0-1.414z"
              : "M10 8.586l4.95-4.95a1 1 0 0 1 1.414 1.414L11.414 10l4.95 4.95a1 1 0 0 1-1.414 1.414L10 11.414l-4.95 4.95a1 1 0 0 1-1.414-1.414L8.586 10l-4.95-4.95a1 1 0 1 1 1.414-1.414L10 8.586z"
          }
          clipRule="evenodd"
        />
      </svg>
    </div>
  );
  
  
  

  return (
    <div className="navbar-nav pt-1">
      <button
        className="flex items-center text-white px-4 py-2 rounded"
        onClick={handleClick}
      >
        <div className="relative h-7 w-7">
          <NearIcon className="h-full w-full shrink-0 focus-visible:outline-0" />
  
          <Icon
            status={hasNearAuth}
            type="top"
            pending={!hasNearAuth && walletConnected}
            onClick={async () => {
              if (hasNearAuth) {
                Cookies.remove("auth");
                clearAuth();
                setHasNearAuth(false);
                return;
              }
            
              if (!walletConnected) {
                walletModal?.show();           
                return;
              }
            
              // Wallet already connected — proceed to NEARAI sign-in
              setReadyToSign(true);
            }}
            setHovered={setHovered}
          />
          <Icon
            status={walletConnected}
            type="bottom"
            onClick={() => {
              if (!selector) return;
              if (walletConnected) {
                wallet?.signOut().then(() => {
                  setWalletConnected(false);
                  navigate(returnUrlToRestoreAfterSignIn());
                });
              } else {
                walletModal?.show();
              }
            }}
            setHovered={setHovered}
          />
  
          {hovered && (
            <div className="absolute left-1/2 -bottom-7 -translate-x-1/2 animate-fade-in-up z-50">
              <span className="rounded bg-gray-800 px-2 py-1 text-xs text-white shadow-md whitespace-nowrap">
                {hovered === "top"
                  ? hasNearAuth
                    ? "NEARAI Connected (Click to Disconnect)"
                    : "NEARAI Not Connected (Click to Connect)"
                  : walletConnected
                    ? "Wallet Connected (Click to Sign Out)"
                    : "Wallet Not Connected (Click to Connect)"}
              </span>
            </div>
          )}
        </div>
      </button>
    </div>
  );
  
}
