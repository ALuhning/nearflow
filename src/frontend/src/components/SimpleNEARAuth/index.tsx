import { useContext, useEffect, useState } from "react";
import { setupWalletSelector, WalletSelector } from "@near-wallet-selector/core";
import { setupModal, WalletSelectorModal } from "@near-wallet-selector/modal-ui";
import { setupHereWallet } from "@near-wallet-selector/here-wallet";
import { setupMeteorWallet } from "@near-wallet-selector/meteor-wallet";
import { Button } from "../../components/ui/button";
import { useNEARChallenge, useNEARSignatureLogin, useNEARAuthEnabled } from "../../controllers/API/queries/auth";
import { useLogout } from "../../controllers/API/queries/auth";
import { AuthContext } from "../../contexts/authContext";
import useAlertStore from "../../stores/alertStore";
import useAuthStore from "../../stores/authStore";
import { useShallow } from "zustand/react/shallow";

interface SimpleNEARAuthProps {
  onLoginStart?: () => void;
  onLoginComplete?: () => void;
  onLoginError?: (error: string) => void;
  onAccountChange?: (accountId: string | null, userExists: boolean, isSuperuser: boolean, stakingMeetsRequirements: boolean) => void;
  hideStakingRequiredMessage?: boolean;
  isSuperuser?: boolean;
  userExists?: boolean;
  nearConfig?: {
    enabled: boolean;
    dev_mode: boolean;
    pool_contract: string;
    min_stake_amount: string;
    superuser: string;
  };
}

export default function SimpleNEARAuth({ 
  onLoginStart,
  onLoginComplete,
  onLoginError,
  onAccountChange,
  hideStakingRequiredMessage = false,
  isSuperuser = false,
  userExists = false,
  nearConfig
}: SimpleNEARAuthProps) {
  const [selector, setSelector] = useState<WalletSelector | null>(null);
  const [modal, setModal] = useState<WalletSelectorModal | null>(null);
  const [loading, setLoading] = useState(false);
  const [nearEnabled, setNearEnabled] = useState(false);
  const [nearConfigState, setNearConfigState] = useState<any>(null);
  const [walletConnected, setWalletConnected] = useState(false);
  const [accountId, setAccountId] = useState<string | null>(null);
  const [stakingRequired, setStakingRequired] = useState(false);
  const [checkingStaking, setCheckingStaking] = useState(false);
  const [stakingAmount, setStakingAmount] = useState<string | null>(null);
  const [stakingToAdd, setStakingToAdd] = useState<string>("25");
  const [stakingInProgress, setStakingInProgress] = useState(false);

  const { login } = useContext(AuthContext);
  const setErrorData = useAlertStore(useShallow((state) => state.setErrorData));

  const challengeMutation = useNEARChallenge();
  const signatureLoginMutation = useNEARSignatureLogin();
  const nearAuthEnabledMutation = useNEARAuthEnabled();
  const logoutMutation = useLogout();

  // Update staking amount input based on current stake
  useEffect(() => {
    if (stakingAmount !== null) {
      const currentStake = parseFloat(stakingAmount);
      const additionalNeeded = Math.max(0, 25 - currentStake);
      
      if (currentStake >= 25) {
        setStakingToAdd("1"); // Default to 1 NEAR for additional staking
      } else {
        setStakingToAdd(additionalNeeded.toFixed(2)); // Set to exact amount needed
      }
    }
  }, [stakingAmount]);

  // Check if NEAR auth is enabled - use props if available, otherwise fetch
  useEffect(() => {
    if (nearConfig) {
      setNearEnabled(nearConfig.enabled);
      setNearConfigState(nearConfig);
    } else {
      nearAuthEnabledMutation.mutate(undefined, {
        onSuccess: (data) => {
          setNearEnabled(data.enabled);
          setNearConfigState(data);
        },
        onError: () => {
          setNearEnabled(false);
        }
      });
    }
  }, [nearConfig]);

  // Initialize wallet selector for detection only
  useEffect(() => {
    if (!nearEnabled) return;

    const initWalletSelector = async () => {
      const walletSelector = await setupWalletSelector({
        network: "mainnet",
        modules: [
          setupHereWallet(),
          setupMeteorWallet(),
        ],
      });

      const walletModal = setupModal(walletSelector, {
        contractId: "",
      });

      setSelector(walletSelector);
      setModal(walletModal);

      // Check wallet connection status
      checkWalletConnection(walletSelector);
      
      // Listen for wallet connection changes
      walletSelector.store.observable.subscribe((state) => {
        const isConnected = state.accounts && state.accounts.length > 0;
        const currentAccount = state.accounts.find(acc => acc.active)?.accountId || null;
        
        console.log("SimpleNEARAuth: Wallet state changed:", { isConnected, currentAccount, previousAccount: accountId });
        
        setWalletConnected(isConnected);
        
        // Check if account actually changed
        if (currentAccount !== accountId) {
          setAccountId(currentAccount);
          
          if (currentAccount) {
            // New account connected or switched - check staking
            console.log("SimpleNEARAuth: Account changed to:", currentAccount);
            checkStakingRequirement(currentAccount);
          } else {
            // Account disconnected
            console.log("SimpleNEARAuth: Account disconnected");
            setStakingRequired(false);
            setStakingAmount(null);
            onAccountChange?.(null, false, false, false);
          }
        }
      });
    };

    initWalletSelector().catch(console.error);
  }, [nearEnabled]);

  const checkWalletConnection = async (walletSelector: WalletSelector) => {
    try {
      const isSignedIn = walletSelector.isSignedIn();
      if (isSignedIn) {
        const wallet = await walletSelector.wallet();
        const accounts = await wallet.getAccounts();
        if (accounts && accounts.length > 0) {
          setWalletConnected(true);
          setAccountId(accounts[0].accountId);
          // Check staking for the connected account
          checkStakingRequirement(accounts[0].accountId);
        }
      }
    } catch (error) {
      console.log("No wallet connected yet");
      setWalletConnected(false);
      setAccountId(null);
    }
  };

  const checkStakingRequirement = async (accountId: string) => {
    if (!accountId) return;
    
    console.log(`SimpleNEARAuth: Starting staking check for account: ${accountId}`);
    setCheckingStaking(true);
    setStakingRequired(false);
    setStakingAmount(null);
    
    try {
      // Call backend to check staking - don't skip based on local isSuperuser state
      // Let the backend determine superuser status
      const response = await fetch(`/api/v1/near-stake-check/${accountId}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      let stakingMeetsRequirements = false;
      let isAccountSuperuser = false;
      
      if (response.ok) {
        const data = await response.json();
        console.log("SimpleNEARAuth: Staking check result:", data);
        
        // Check if this account is a superuser based on the API response
        isAccountSuperuser = data.superuser || false;
        
        if (data.meets_requirements || isAccountSuperuser) {
          setStakingRequired(false);
          setStakingAmount(data.current_stake);
          stakingMeetsRequirements = true;
          console.log(`SimpleNEARAuth: Account ${accountId} meets requirements. Superuser: ${isAccountSuperuser}`);
        } else {
          setStakingRequired(true);
          setStakingAmount(data.current_stake || "0");
          stakingMeetsRequirements = false;
          console.log(`SimpleNEARAuth: Account ${accountId} does not meet staking requirements. Current stake: ${data.current_stake}`);
        }
      } else {
        console.log("SimpleNEARAuth: Failed to check staking, assuming requirements not met");
        setStakingRequired(true);
        stakingMeetsRequirements = false;
      }
      
      // Check if user already exists
      let userExistsForAccount = false;
      try {
        const userExistsResponse = await fetch(`/api/v1/check-user-exists/${accountId}`);
        if (userExistsResponse.ok) {
          const userData = await userExistsResponse.json();
          userExistsForAccount = userData.exists;
          console.log(`SimpleNEARAuth: User exists check for ${accountId}: ${userExistsForAccount}`);
        }
      } catch (error) {
        console.error("SimpleNEARAuth: Error checking user existence:", error);
      }
      
      // Always notify parent component about the account state
      console.log(`SimpleNEARAuth: Calling onAccountChange with:`, { 
        accountId, 
        userExists: userExistsForAccount, 
        isSuperuser: isAccountSuperuser, 
        stakingMeetsRequirements 
      });
      onAccountChange?.(accountId, userExistsForAccount, isAccountSuperuser, stakingMeetsRequirements);
      
    } catch (error) {
      console.error("SimpleNEARAuth: Error checking staking:", error);
      // Assume staking required if check fails
      setStakingRequired(true);
      // Still notify parent about account change, even if checks failed
      console.log(`SimpleNEARAuth: Error occurred, calling onAccountChange with default values for ${accountId}`);
      onAccountChange?.(accountId, false, false, false);
    } finally {
      setCheckingStaking(false);
    }
  };

  const handleConnectWallet = async () => {
    if (!modal) return;
    
    setLoading(true);
    try {
      modal.show();
    } catch (error) {
      console.error("Error showing wallet modal:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleCreateAccount = async () => {
    if (!selector || !accountId) {
      setErrorData({
        title: "NEAR Wallet Error",
        list: ["No wallet connected"],
      });
      return;
    }

    try {
      setLoading(true);
      onLoginStart?.();

      // Get the connected wallet (no popup needed - already connected)
      const wallet = await selector.wallet();
      
      console.log(`Creating NearFlow account for: ${accountId}`);

      // Step 1: Get challenge from backend
      console.log("Getting challenge from backend...");
      const challengeResponse = await new Promise<any>((challengeResolve, challengeReject) => {
        challengeMutation.mutate(
          { near_account_id: accountId },
          {
            onSuccess: challengeResolve,
            onError: challengeReject,
          }
        );
      });

      console.log("Challenge received:", challengeResponse);

      // Step 2: Sign the message using existing wallet connection
      const message = challengeResponse.message;
      const recipient = challengeResponse.recipient;
      const nonce = Buffer.from(challengeResponse.challenge, 'base64');
      
      // Check if wallet supports message signing
      if (!wallet.signMessage) {
        throw new Error("This wallet doesn't support message signing. Please try a different wallet.");
      }
      
      console.log("Signing message with connected wallet...");
      
      // Sign the message - using existing connection, no popup needed
      let signResult;
      try {
        signResult = await wallet.signMessage({
          message,
          nonce,
          recipient,
        });
      } catch (signError: any) {
        console.error("Message signing error:", signError);
        
        // Handle Ledger-specific errors
        if (signError?.message?.includes("Ledger") || 
            signError?.message?.includes("UNKNOWN_ERROR") ||
            signError?.message?.includes("0xb005") ||
            signError?.message?.includes("unavailable")) {
          throw new Error("Ledger Hardware Wallet Error: Please ensure your Ledger device is connected, unlocked, and the NEAR app is open and up to date. If the problem persists, try using a different wallet or check your Ledger's firmware version.");
        }
        
        // Handle general signing errors
        if (signError?.message?.includes("User rejected")) {
          throw new Error("Transaction was cancelled by user");
        }
        
        // Re-throw with original error if not a known case
        throw signError;
      }
      
      if (!signResult) {
        throw new Error("Message signing failed - no result returned from wallet");
      }
      
      console.log("Message signed successfully");
      console.log("Sign result:", signResult);
      
      // Extract signature and public key
      const signature = (signResult as any).signature;
      const accounts = await wallet.getAccounts();
      
      // Get the public key that was actually used for signing
      // This should be the FullAccess key from the signResult
      let publicKey = (signResult as any).publicKey;
      
      console.log("Available accounts:", accounts);
      console.log("Public key from sign result:", publicKey);
      console.log("Wallet type:", (selector as any)?.store?.getState()?.selectedWalletId);
      
      const walletId = (selector as any)?.store?.getState()?.selectedWalletId;
      
      // Handle hardware wallet special case (applies to MeteorWallet/HereWallet + Ledger)
      if (!publicKey) {
        console.log("No publicKey in sign result - checking if this is a hardware wallet flow");
        console.log("Will attempt backend verification with signature to determine the correct key");
        
        // For hardware wallets, we'll send a special marker to the backend
        // The backend will try to verify the signature against all available keys for the account
        publicKey = "LEDGER_AUTO_DETECT";
      } else if (!publicKey) {
        // For other wallets, fall back to account public key
        console.warn("Public key not provided in sign result, using account's primary key");
        
        if (accounts && accounts.length > 0) {
          publicKey = accounts[0].publicKey;
          console.log("Using primary account public key:", publicKey);
        } else {
          throw new Error("No account information available to determine public key. Please try reconnecting your wallet.");
        }
      }
      
      if (!signature || !publicKey) {
        throw new Error("Invalid signature or public key from message signing");
      }
      
      // Ensure the public key has the ed25519: prefix (unless it's the special marker)
      let fullPublicKey;
      if (publicKey === "LEDGER_AUTO_DETECT") {
        fullPublicKey = publicKey; // Keep the marker as-is
      } else {
        fullPublicKey = publicKey.startsWith('ed25519:') ? publicKey : `ed25519:${publicKey}`;
      }
      
      console.log("Raw public key from wallet:", publicKey);
      console.log("Full public key for verification:", fullPublicKey);
      console.log("Signature:", signature);
      
      // Step 3: Submit signature to backend
      console.log("Submitting signature to backend...");
      console.log("Login payload:", {
        near_account_id: accountId,
        public_key: fullPublicKey,
        signature: signature,
        challenge: challengeResponse.challenge,
        message: challengeResponse.message,
        recipient: challengeResponse.recipient,
      });
      
      const loginResponse = await new Promise<any>((loginResolve, loginReject) => {
        signatureLoginMutation.mutate(
          {
            near_account_id: accountId,
            public_key: fullPublicKey,
            signature: signature,
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
      
      console.log("Login successful!");
      
      // Login successful
      login(loginResponse.access_token, "near-auth", loginResponse.refresh_token);
      onLoginComplete?.();

    } catch (error: any) {
      console.error("NEAR login error:", error);
      console.error("Error details:", {
        message: error?.message,
        response: error?.response?.data,
        status: error?.response?.status,
        accountId: accountId
      });
      
      // Provide more specific error messages
      let errorMessage = error?.response?.data?.detail || error?.message || "NEAR authentication failed";
      
      // Check for specific error types
      if (errorMessage.includes("Ledger Hardware Wallet Error")) {
        // Keep the detailed Ledger error message as-is
        errorMessage = error.message;
      } else if (errorMessage.includes("Public key ownership verification failed") || 
                 errorMessage.includes("Ownership verification failed") ||
                 errorMessage.includes("Invalid key or accountId") ||
                 errorMessage.includes("Failed to auto-detect signing key")) {
        console.log("Ownership verification failed - this is common with hardware wallets");
        
        // Check if this is a hardware wallet scenario
        const walletId = (selector as any)?.store?.getState()?.selectedWalletId;
        console.log("Current wallet ID:", walletId);
        
        errorMessage = "Hardware Wallet Authentication Issue: Your hardware wallet is connected but there's a technical issue with key verification. This can happen with Ledger devices.\n\nRecommended solutions:\n• Try disconnecting and reconnecting your hardware wallet\n• Ensure your Ledger has the latest firmware and NEAR app version\n• Use MeteorWallet or HereWallet for better hardware wallet support\n• If the issue persists, try using a hot wallet for authentication";
        
      } else if (errorMessage.includes("Signature verification failed")) {
        errorMessage = "Authentication failed: Invalid signature. Please try connecting your wallet again.";
      } else if (errorMessage.includes("does not meet minimum staking requirements")) {
        errorMessage = "Staking requirement not met. Please stake at least 25 NEAR with vitalpoint.pool.near to access NearFlow.";
      } else if (errorMessage.includes("doesn't support message signing")) {
        errorMessage = "This wallet doesn't support the required authentication method. Please try using MeteorWallet, HereWallet, or another compatible wallet.";
      } else if (errorMessage.includes("User rejected") || errorMessage.includes("cancelled")) {
        errorMessage = "Authentication was cancelled. Please try again and approve the signature request.";
      }
      
      setErrorData({
        title: "NEAR Authentication Failed",
        list: [errorMessage],
      });
      
      onLoginError?.(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  const handleStakeToVitalPoint = async () => {
    if (!selector || !accountId) {
      setErrorData({
        title: "Staking Error",
        list: ["No wallet connected"],
      });
      return;
    }

    try {
      setStakingInProgress(true);
      
      const wallet = await selector.wallet();
      const stakeAmount = parseFloat(stakingToAdd);
      
      if (stakeAmount < 1) {
        throw new Error("Minimum stake amount is 1 NEAR");
      }

      console.log(`Staking ${stakeAmount} NEAR to vitalpoint.pool.near`);

      // Convert NEAR to yoctoNEAR properly (avoid scientific notation)
      const yoctoNearAmount = BigInt(Math.floor(stakeAmount * 1000000)) * BigInt("1000000000000000000"); // 1e24 yoctoNEAR per NEAR
      
      console.log(`Converting ${stakeAmount} NEAR to ${yoctoNearAmount.toString()} yoctoNEAR`);

      // Create staking transaction
      const transaction = {
        receiverId: "vitalpoint.pool.near",
        actions: [
          {
            type: "FunctionCall" as const,
            params: {
              methodName: "deposit_and_stake",
              args: {},
              gas: "125000000000000", // 125 TGas
              deposit: yoctoNearAmount.toString(), // Convert to string to avoid BigInt issues
            },
          },
        ],
      };

      console.log("Sending staking transaction:", transaction);

      // Send the staking transaction
      const result = await wallet.signAndSendTransaction(transaction);
      
      console.log("Staking transaction result:", result);
      
      // Show success message
      setErrorData({
        title: "Staking Successful!",
        list: [`Successfully staked ${stakeAmount} NEAR to vitalpoint.pool.near`],
      });
      
      // Wait a moment for the transaction to be processed, then re-check staking
      setTimeout(() => {
        checkStakingRequirement(accountId);
      }, 5000); // Increased to 5 seconds for transaction processing
      
    } catch (error: any) {
      console.error("Staking error:", error);
      
      const errorMessage = error?.message || "Failed to stake NEAR tokens";
      
      setErrorData({
        title: "Staking Failed",
        list: [errorMessage],
      });
    } finally {
      setStakingInProgress(false);
    }
  };

  const handleSwitchAccount = async () => {
    if (!selector) return;
    
    try {
      setLoading(true);
      
      // First, logout from backend to clear session and cookies
      console.log("Logging out from backend authentication...");
      await new Promise<void>((resolve, reject) => {
        logoutMutation.mutate(undefined, {
          onSuccess: () => {
            console.log("Backend logout successful");
            resolve();
          },
          onError: (error) => {
            console.error("Backend logout error:", error);
            // Continue with wallet logout even if backend logout fails
            resolve();
          }
        });
      });
      
      // Sign out from current wallet
      console.log("Signing out from NEAR wallet...");
      const wallet = await selector.wallet();
      await wallet.signOut();
      
      // Reset local component state
      setWalletConnected(false);
      setAccountId(null);
      setStakingRequired(false);
      setStakingAmount(null);
      setCheckingStaking(false);
      
      // Notify parent that account was reset
      onAccountChange?.(null, false, false, false);
      
      console.log("Account switch completed - state reset");
      
      // Show wallet selection modal for new account
      if (modal) {
        setTimeout(() => {
          modal.show();
        }, 500); // Small delay to ensure cleanup
      }
      
    } catch (error) {
      console.error("Error switching account:", error);
      setErrorData({
        title: "Account Switch Error",
        list: ["Failed to switch NEAR account"],
      });
    } finally {
      setLoading(false);
    }
  };

  // Re-check staking requirements when isSuperuser prop changes (for account switching)
  useEffect(() => {
    if (accountId) {
      console.log(`isSuperuser prop changed to ${isSuperuser} for account ${accountId}, re-checking staking requirements`);
      checkStakingRequirement(accountId);
    }
  }, [isSuperuser, accountId]);

  if (!nearEnabled) {
    return null;
  }

  // Helper function to format stake amount
  const formatStakeAmount = (amount: string | null): string => {
    if (!amount) return "0";
    const numAmount = parseFloat(amount);
    if (numAmount < 0.001) return numAmount.toFixed(6);
    if (numAmount < 1) return numAmount.toFixed(3);
    return numAmount.toFixed(2);
  };

  // Helper function to calculate additional stake needed
  const getAdditionalStakeNeeded = (current: string | null): number => {
    const currentAmount = parseFloat(current || "0");
    return Math.max(0, 25 - currentAmount);
  };

  // Show staking requirement page if needed (only for non-superusers)
  if (stakingRequired && accountId && !isSuperuser) {
    const currentStake = parseFloat(stakingAmount || "0");
    const additionalNeeded = getAdditionalStakeNeeded(stakingAmount);
    const formattedCurrent = formatStakeAmount(stakingAmount);
    
    return (
      <div className="w-full space-y-4">
        {!hideStakingRequiredMessage && (
          <div className="text-center bg-yellow-50 p-4 rounded-md border border-yellow-200">
            <div className="font-medium text-yellow-800 mb-2">🔒 Staking Required</div>
            <div className="text-yellow-700 text-sm space-y-2">
              <div>Account <strong>{accountId}</strong> needs to stake NEAR to access NearFlow.</div>
              <div>Current stake: <strong>{formattedCurrent} NEAR</strong></div>
              {currentStake >= 25 ? (
                <div>You have enough stake! You can stake more if you'd like, or proceed with account creation.</div>
              ) : (
                <div>You need <strong>{additionalNeeded.toFixed(2)} more NEAR</strong> (minimum 25 NEAR total) with <strong>vitalpoint.pool.near</strong></div>
              )}
            </div>
          </div>
        )}
        
        {/* Direct staking option */}
        <div className="bg-white p-4 rounded-md border border-gray-200 space-y-3">
          <div className="font-medium text-gray-800">
            {currentStake >= 25 ? "Stake More (Optional)" : "Stake to Meet Requirement"}
          </div>
          <div className="flex space-x-2">
            <div className="flex-1">
              <label className="block text-xs text-gray-600 mb-1">
                NEAR Amount {currentStake < 25 && `(minimum ${additionalNeeded.toFixed(2)} more)`}
              </label>
              <input
                type="number"
                value={stakingToAdd}
                onChange={(e) => setStakingToAdd(e.target.value)}
                min={currentStake >= 25 ? "1" : additionalNeeded.toFixed(2)}
                step="0.1"
                className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
                placeholder={currentStake >= 25 ? "1" : additionalNeeded.toFixed(2)}
                disabled={stakingInProgress}
              />
            </div>
          </div>
          <Button
            onClick={handleStakeToVitalPoint}
            disabled={stakingInProgress || (currentStake < 25 && parseFloat(stakingToAdd) < additionalNeeded)}
            className="w-full"
            type="button"
          >
            {stakingInProgress ? "Staking..." : `Stake ${stakingToAdd} NEAR to vitalpoint.pool.near`}
          </Button>
          {currentStake < 25 && (
            <div className="text-xs text-yellow-600 text-center">
              Minimum {additionalNeeded.toFixed(2)} NEAR needed to meet the 25 NEAR requirement
            </div>
          )}
        </div>
        
        <Button
          onClick={() => checkStakingRequirement(accountId)}
          variant="outline"
          className="w-full"
          type="button"
          disabled={checkingStaking}
        >
          {checkingStaking ? "Checking..." : "Check Staking Again"}
        </Button>
        
        {/* Switch Account Button */}
        <Button
          onClick={handleSwitchAccount}
          variant="outline"
          className="w-full"
          type="button"
          disabled={loading}
        >
          {loading ? "Switching..." : "Switch NEAR Account"}
        </Button>
        
        <div className="text-xs text-muted-foreground text-center space-y-2">
          <div>After staking, click "Check Staking Again" to verify</div>
          <div>Or switch to a different NEAR account that already meets the requirements</div>
          <div>Staking helps secure the NEAR network and provides access to NearFlow</div>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full space-y-3">
      {!walletConnected ? (
        // Step 1: Connect wallet
        <>
          <Button
            onClick={handleConnectWallet}
            disabled={loading || !selector}
            className="w-full"
            variant="outline"
            type="button"
          >
            {loading ? "Connecting..." : "Connect NEAR Wallet"}
          </Button>
          <div className="text-xs text-muted-foreground text-center">
            <div>{isSuperuser ? 'Connect your admin NEAR wallet to continue' : 'First, connect your NEAR wallet to continue'}</div>
          </div>
        </>
      ) : checkingStaking ? (
        // Step 2a: Checking staking requirements
        <>
          <div className="text-sm text-center bg-blue-50 p-3 rounded-md border border-blue-200">
            <div className="font-medium text-blue-800">🔍 {isSuperuser ? 'Verifying Admin Access' : 'Checking Staking Requirements'}</div>
            <div className="text-blue-600 text-xs mt-1">{accountId}</div>
          </div>
          <div className="text-xs text-muted-foreground text-center">
            <div>{isSuperuser ? 'Verifying your admin status...' : 'Verifying your NEAR staking status...'}</div>
          </div>
        </>
      ) : (
        // Step 2b: Create NearFlow account (staking verified or superuser)
        <>
          <div className={`text-sm text-center p-3 rounded-md border ${isSuperuser ? 'bg-purple-50 border-purple-200' : 'bg-green-50 border-green-200'}`}>
            <div className={`font-medium ${isSuperuser ? 'text-purple-800' : 'text-green-800'}`}>
              ✓ {isSuperuser ? 'Wallet Connected & Admin Access Verified' : 'Wallet Connected & Staking Verified'}
            </div>
            <div className={`text-xs mt-1 ${isSuperuser ? 'text-purple-600' : 'text-green-600'}`}>{accountId}</div>
            {stakingAmount && !isSuperuser && <div className="text-green-600 text-xs">Stake: {formatStakeAmount(stakingAmount)} NEAR</div>}
            {isSuperuser && <div className="text-purple-600 text-xs">👑 Administrator</div>}
          </div>
          <Button
            onClick={handleCreateAccount}
            disabled={loading}
            className="w-full"
            type="button"
          >
            {loading ? 
              (userExists ? "Signing In..." : "Creating Account...") : 
              (userExists ? "Sign In to NearFlow" : "Create NearFlow Account")
            }
          </Button>
          <div className="text-xs text-muted-foreground text-center">
            <div>{isSuperuser ? 'Sign the authentication message to access admin features' : `Sign the authentication message to ${userExists ? 'sign in to' : 'create'} your account`}</div>
          </div>
        </>
      )}
      {walletConnected && (
        <div className="text-center">
          <Button
            onClick={handleSwitchAccount}
            disabled={loading}
            variant="outline"
            className="w-full"
            type="button"
          >
            {loading ? "Switching..." : "Switch NEAR Account"}
          </Button>
          <div className="text-xs text-muted-foreground mt-2">
            <div>{isSuperuser ? 'Switch to a different admin NEAR account if needed' : 'Connect a different NEAR account if needed'}</div>
          </div>
        </div>
      )}
    </div>
  );
}
