import { useContext, useEffect, useState } from "react";
import { setupWalletSelector, WalletSelector } from "@near-wallet-selector/core";
import { setupModal, WalletSelectorModal } from "@near-wallet-selector/modal-ui";
import { setupHereWallet } from "@near-wallet-selector/here-wallet";
import { setupMeteorWallet } from "@near-wallet-selector/meteor-wallet";
import { setupSender } from "@near-wallet-selector/sender";
// Skip Ledger wallet for now to avoid Buffer issues
// import { setupLedger } from "@near-wallet-selector/ledger";
import { setupWelldoneWallet } from "@near-wallet-selector/welldone-wallet";
import * as nearAPI from "near-api-js";
import { Button } from "../../components/ui/button";
import { useNEARChallenge, useNEARSignatureLogin, useNEARAuthEnabled } from "../../controllers/API/queries/auth";
import { AuthContext } from "../../contexts/authContext";
import useAlertStore from "../../stores/alertStore";
import { useShallow } from "zustand/react/shallow";
import { Buffer } from "buffer";

interface NEARAuthComponentProps {
  onLoginStart?: () => void;
  onLoginComplete?: () => void;
  onLoginError?: (error: string) => void;
}

export default function NEARAuthComponent({ 
  onLoginStart, 
  onLoginComplete, 
  onLoginError 
}: NEARAuthComponentProps) {
  const [selector, setSelector] = useState<WalletSelector | null>(null);
  const [modal, setModal] = useState<WalletSelectorModal | null>(null);
  const [loading, setLoading] = useState(false);
  const [nearEnabled, setNearEnabled] = useState(false);
  const [nearConfig, setNearConfig] = useState<any>(null);

  const { login } = useContext(AuthContext);
  const setErrorData = useAlertStore(useShallow((state) => state.setErrorData));

  const challengeMutation = useNEARChallenge();
  const signatureLoginMutation = useNEARSignatureLogin();
  const nearAuthEnabledMutation = useNEARAuthEnabled();

  // Check if NEAR auth is enabled
  useEffect(() => {
    nearAuthEnabledMutation.mutate(undefined, {
      onSuccess: (data) => {
        setNearEnabled(data.enabled);
        setNearConfig(data);
      },
      onError: () => {
        setNearEnabled(false);
      }
    });
  }, []);

  // Initialize wallet selector
  useEffect(() => {
    if (!nearEnabled) return;

    const initWalletSelector = async () => {
      try {
        const selector = await setupWalletSelector({
          network: "mainnet",
          modules: [
            setupHereWallet(),
            setupMeteorWallet(),
            setupSender(),
            // setupLedger(), // Commented out to avoid Buffer issues
            setupWelldoneWallet(),
          ],
        });

        const modal = setupModal(selector, {
          contractId: nearConfig?.pool_contract || "vitalpoint.pool.near",
        });

        setSelector(selector);
        setModal(modal);
      } catch (error) {
        console.error("Failed to initialize wallet selector:", error);
      }
    };

    initWalletSelector();
  }, [nearEnabled, nearConfig]);

  const handleNEARLogin = async () => {
    if (!selector || !modal) {
      setErrorData({
        title: "NEAR Authentication Error",
        list: ["Wallet selector not initialized"],
      });
      return;
    }

    try {
      setLoading(true);
      onLoginStart?.();

      // Show wallet selection modal
      modal.show();

      // Wait for wallet connection
      return new Promise<void>((resolve, reject) => {
        const subscription = selector.on("signedIn", async (e) => {
          try {
            const accountId = e.accounts[0]?.accountId;
            
            if (!accountId) {
              throw new Error("No account ID found");
            }

            // Step 1: Get challenge from backend
            const challengeResponse = await new Promise<any>((challengeResolve, challengeReject) => {
              challengeMutation.mutate(
                { near_account_id: accountId },
                {
                  onSuccess: challengeResolve,
                  onError: challengeReject,
                }
              );
            });

            // Sign the message using NEAR wallet
            const wallet = await selector.wallet();
            
            // Sign the message using NEAR wallet
            const signResult = await wallet.signMessage({
              message: challengeResponse.message,
              recipient: challengeResponse.recipient,
              nonce: Buffer.from(challengeResponse.challenge, 'base64'),
            });

            if (!signResult) {
              throw new Error("Failed to sign message");
            }

            // Step 3: Submit signature to backend
            const loginResponse = await new Promise<any>((loginResolve, loginReject) => {
              signatureLoginMutation.mutate(
                {
                  near_account_id: accountId,
                  public_key: signResult.publicKey,
                  signature: signResult.signature,
                  challenge: challengeResponse.challenge,
                  message: challengeResponse.message,
                  recipient: challengeResponse.recipient,
                },
                {
                  onSuccess: loginResolve,
                  onError: loginReject,
                }
              );
            });

            // Login successful
            login(loginResponse.access_token, "near-auth", loginResponse.refresh_token);
            onLoginComplete?.();

            // Show success message with stake info
            if (loginResponse.stake_amount) {
              setErrorData({
                title: "Login Successful",
                list: [`Welcome! Your stake: ${loginResponse.stake_amount} NEAR`],
              });
            }

            subscription.remove();
            resolve();
          } catch (error: any) {
            subscription.remove();
            reject(error);
          } finally {
            setLoading(false);
            modal?.hide();
          }
        });

        // Handle modal close without signing in
        const modalCloseSubscription = selector.on("signedOut", () => {
          subscription.remove();
          modalCloseSubscription.remove();
          setLoading(false);
          reject(new Error("User cancelled wallet connection"));
        });
      });

    } catch (error: any) {
      console.error("NEAR login error:", error);
      const errorMessage = error?.response?.data?.detail || error?.message || "NEAR authentication failed";
      
      setErrorData({
        title: "NEAR Authentication Failed",
        list: [errorMessage],
      });
      
      onLoginError?.(errorMessage);
      setLoading(false);
      modal?.hide();
    }
  };

  // Don't render if NEAR auth is not enabled
  if (!nearEnabled) {
    return null;
  }

  return (
    <div className="w-full">
      <div className="mb-4 text-center">
        <div className="relative">
          <div className="absolute inset-0 flex items-center">
            <span className="w-full border-t" />
          </div>
          <div className="relative flex justify-center text-xs uppercase">
            <span className="bg-muted px-2 text-muted-foreground">
              Or continue with
            </span>
          </div>
        </div>
      </div>
      
      <Button 
        className="w-full"
        variant="outline" 
        type="button"
        onClick={handleNEARLogin}
        disabled={loading || !selector}
      >
        {loading ? (
          <>
            <div className="mr-2 h-4 w-4 animate-spin rounded-full border-2 border-gray-300 border-t-gray-600" />
            Connecting to NEAR...
          </>
        ) : (
          <>
            <svg className="mr-2 h-4 w-4" viewBox="0 0 24 24" fill="currentColor">
              <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
            </svg>
            Sign in with NEAR Wallet
          </>
        )}
      </Button>
      
      {nearConfig && (
        <div className="mt-2 text-xs text-muted-foreground text-center">
          Requires minimum {nearConfig.min_stake_amount} NEAR staked in {nearConfig.pool_contract}
        </div>
      )}
    </div>
  );
}
