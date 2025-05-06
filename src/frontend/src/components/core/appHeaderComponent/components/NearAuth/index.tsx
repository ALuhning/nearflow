import { useEffect, useState } from "react";
import { useNavigate } from "react-router";
import NearIcon from "@/assets/near-icon.svg?react";
import {
  returnUrlToRestoreAfterSignIn,
  generateNonce,
  createAuthUrl,
} from "@/utils/auth";
import { useQueryClient } from "@tanstack/react-query";
import { config } from "@/config";
import { useWalletStore } from "@/stores/walletStore";
import { useShallow } from "zustand/react/shallow";
import { useSignAuthUrlMessage } from "./SignIn";
import { usePostGlobalVariables } from "@/controllers/API/queries/variables/use-post-global-variables";
import { usePatchGlobalVariables } from "@/controllers/API/queries/variables/use-patch-global-variables";
import { useGetGlobalVariables } from "@/controllers/API/queries/variables";
import {
  useSessionQuery,
  useSignOutMutation,
} from "@/controllers/API/queries/auth/use-post-login-user-near";
import { useDeleteGlobalVariables } from "@/controllers/API/queries/variables";

export default function NearAuthIcon() {
  const { MESSAGE, RECIPIENT } = config;
  const navigate = useNavigate();
  const selector = useWalletStore((state) => state.selector);
  const walletAccount = useWalletStore(useShallow((s) => s.account));
  const wallet = useWalletStore(useShallow((state) => state.wallet));
  const walletModal = useWalletStore(useShallow((state) => state.modal));
  const queryClient = useQueryClient();

  const { signFromAuthUrl } = useSignAuthUrlMessage();
  const { data: variables } = useGetGlobalVariables();
  const { mutateAsync: postGlobalVar } = usePostGlobalVariables();
  const { mutateAsync: patchGlobalVar } = usePatchGlobalVariables();
  const { mutateAsync: signOutNearAuth } = useSignOutMutation();
  const { mutateAsync: deleteGlobalVar } = useDeleteGlobalVariables();
  const { data: session, refetch: refetchSession } = useSessionQuery();

  const [walletConnected, setWalletConnected] = useState(false);
  const [readyToSign, setReadyToSign] = useState(false);
  const [toggle, setToggle] = useState(false);
  const [hasNearAuth, setHasNearAuth] = useState(false);
  const [hovered, setHovered] = useState<"top" | "bottom" | null>(null);

  const authUrl = createAuthUrl(MESSAGE, RECIPIENT, generateNonce());

  const tryUpsertGlobalVar = async (cookieObject: any) => {
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
    queryClient.invalidateQueries({ queryKey: ["variables"] as const });
    return true
  };

  // Debounced session effect to prevent premature variable creation
  useEffect(() => {
    const timeout = setTimeout(() => {
      if (!session) return;

      const hasAuthFields =
        session.account_id &&
        session.signature &&
        session.public_key &&
        session.message &&
        session.nonce &&
        session.recipient;

      if (hasAuthFields) {
        setHasNearAuth(true);
        const cookieObject = {
          account_id: session.account_id,
          signature: session.signature,
          public_key: session.public_key,
          message: session.message,
          nonce: session.nonce,
          recipient: session.recipient,
          callback_url: session.callback_url,
          on_behalf_of: session.on_behalf_of,
        };
        tryUpsertGlobalVar(cookieObject);
        queryClient.invalidateQueries({ queryKey: ["variables"] as const });
      } else {
        setHasNearAuth(false);
        queryClient.invalidateQueries({ queryKey: ["variables"] as const });
      }
    }, 150); // debounce duration

    return () => clearTimeout(timeout);
  }, [session]);

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
    queryClient.invalidateQueries({ queryKey: ["variables"] as const });
    if (!readyToSign || !walletConnected) return;
    (async () => {
      try {
        await signFromAuthUrl(authUrl);
        if(session){
        const cookieObject = {
          account_id: session.account_id,
          signature: session.signature,
          public_key: session.public_key,
          message: session.message,
          nonce: session.nonce,
          recipient: session.recipient,
          callback_url: session.callback_url,
          on_behalf_of: session.on_behalf_of,
        };
      
        const success = await tryUpsertGlobalVar(cookieObject);
          if (success) {
            setHasNearAuth(true);
            queryClient.invalidateQueries({ queryKey: ["variables"] as const });
          }
        }
        
      } catch {
        setHasNearAuth(false);
        const existing = variables?.find((v) => v.name === "NEARAI");
        if (existing) {
          await deleteGlobalVar({ id: existing.id });
          queryClient.invalidateQueries({ queryKey: ["variables"] as const });
        }
      } finally {
        setReadyToSign(false);
      }
    })();
  }, [readyToSign, session]);

  useEffect(() => {
    if (!toggle || !walletAccount) return;
    (async () => {
      try {
        await wallet?.signOut();
        setWalletConnected(false);
        setHasNearAuth(false);
        const existing = variables?.find((v) => v.name === "NEARAI");
        if (existing) {
          await deleteGlobalVar({ id: existing.id });
          queryClient.invalidateQueries({ queryKey: ["variables"] as const });
        }
        navigate(returnUrlToRestoreAfterSignIn());
      } finally {
        setToggle(false);
      }
    })();
  }, [toggle, walletAccount]);

  const handleClick = () => setToggle((t) => !t);

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
            ? "text-green-500" +
              (type === "top" && !hasNearAuth && walletConnected
                ? " animate-pulse"
                : "")
            : "text-red-500"
        }`}
        fill="currentColor"
        viewBox="0 0 20 20"
      >
        <path
          fillRule="evenodd"
          d={
            pending
              ? "M10 2a8 8 0 100 16 8 8 0 000-16zm1 4H9v5h4v-2h-3V6z"
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
                await signOutNearAuth();
                setHasNearAuth(false);
                const existing = variables?.find((v) => v.name === "NEARAI");
                if (existing) {
                  await deleteGlobalVar({ id: existing.id });
                  queryClient.invalidateQueries({
                    queryKey: ["variables"] as const,
                  });
                }
              } else if (!walletConnected) {
                walletModal?.show();
              } else {
                setReadyToSign(true);
              }
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
