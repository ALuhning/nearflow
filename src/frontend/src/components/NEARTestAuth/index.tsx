import { useContext, useState } from "react";
import { Button } from "../../components/ui/button";
import { Input } from "../../components/ui/input";
import { useNEARChallenge, useNEARSignatureLogin } from "../../controllers/API/queries/auth";
import { AuthContext } from "../../contexts/authContext";
import useAlertStore from "../../stores/alertStore";
import { useShallow } from "zustand/react/shallow";

interface NEARTestAuthProps {
  onLoginStart?: () => void;
  onLoginComplete?: () => void;
  onLoginError?: (error: string) => void;
}

export default function NEARTestAuth({ 
  onLoginStart, 
  onLoginComplete, 
  onLoginError 
}: NEARTestAuthProps) {
  const [accountId, setAccountId] = useState("vitalpoint.near");
  const [loading, setLoading] = useState(false);

  const { login } = useContext(AuthContext);
  const setErrorData = useAlertStore(useShallow((state) => state.setErrorData));

  const challengeMutation = useNEARChallenge();
  const signatureLoginMutation = useNEARSignatureLogin();

  const handleTestNEARLogin = async () => {
    if (!accountId) {
      setErrorData({
        title: "NEAR Authentication Error",
        list: ["Please enter a NEAR account ID"],
      });
      return;
    }

    try {
      setLoading(true);
      onLoginStart?.();

      // Step 1: Get challenge from backend
      const challengeResponse = await new Promise<any>((resolve, reject) => {
        challengeMutation.mutate(
          { near_account_id: accountId },
          {
            onSuccess: resolve,
            onError: reject,
          }
        );
      });

      console.log("Challenge received:", challengeResponse);

      // Step 2: For testing, use mock signature data
      // In real implementation, this would come from wallet signing
      const mockSignature = btoa(`mock_signature_${Date.now()}`);
      const mockPublicKey = "ed25519:MockPublicKeyForTesting123456789";

      // Step 3: Submit signature to backend
      const loginResponse = await new Promise<any>((resolve, reject) => {
        signatureLoginMutation.mutate(
          {
            near_account_id: accountId,
            public_key: mockPublicKey,
            signature: mockSignature,
            challenge: challengeResponse.challenge,
            message: challengeResponse.message,
            recipient: challengeResponse.recipient,
          },
          {
            onSuccess: resolve,
            onError: reject,
          }
        );
      });

      // Login successful
      login(loginResponse.access_token, "near-auth", loginResponse.refresh_token);
      onLoginComplete?.();

      // Show success message with stake info
      setErrorData({
        title: "NEAR Login Test",
        list: [
          "Challenge/response flow completed successfully!",
          `Account: ${accountId}`,
          loginResponse.stake_amount ? `Stake: ${loginResponse.stake_amount} NEAR` : "No stake info"
        ],
      });

    } catch (error: any) {
      console.error("NEAR login error:", error);
      const errorMessage = error?.response?.data?.detail || error?.message || "NEAR authentication failed";
      
      setErrorData({
        title: "NEAR Authentication Test Failed",
        list: [errorMessage],
      });
      
      onLoginError?.(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="w-full">
      <div className="mb-4 text-center">
        <div className="relative">
          <div className="absolute inset-0 flex items-center">
            <span className="w-full border-t" />
          </div>
          <div className="relative flex justify-center text-xs uppercase">
            <span className="bg-muted px-2 text-muted-foreground">
              NEAR Authentication Test
            </span>
          </div>
        </div>
      </div>
      
      <div className="mb-3 w-full">
        <Input
          type="text"
          value={accountId}
          onChange={(e) => setAccountId(e.target.value)}
          placeholder="Enter NEAR account ID (e.g., user.near)"
          className="w-full"
        />
      </div>
      
      <Button 
        className="w-full"
        variant="outline" 
        type="button"
        onClick={handleTestNEARLogin}
        disabled={loading || !accountId}
      >
        {loading ? (
          <>
            <div className="mr-2 h-4 w-4 animate-spin rounded-full border-2 border-gray-300 border-t-gray-600" />
            Testing NEAR Auth...
          </>
        ) : (
          <>
            <svg className="mr-2 h-4 w-4" viewBox="0 0 24 24" fill="currentColor">
              <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
            </svg>
            Test NEAR Challenge/Response Flow
          </>
        )}
      </Button>
      
      <div className="mt-2 text-xs text-muted-foreground text-center">
        This tests the backend NEAR authentication endpoints without requiring wallet connection
      </div>
    </div>
  );
}
