import { useMutation, UseMutationResult } from "@tanstack/react-query";
import { api } from "../../api";

// Types for NEAR challenge/response authentication
export interface NEARChallengeRequest {
  near_account_id: string;
}

export interface NEARChallengeResponse {
  challenge: string;
  message: string;
  recipient: string;
}

export interface NEARSignatureLogin {
  near_account_id: string;
  public_key: string;
  signature: string;
  challenge: string;
  message: string;
  recipient: string;
}

export interface NEARLoginResponse {
  access_token: string;
  refresh_token: string;
  token_type: string;
  user_created: boolean;
  stake_amount?: string;
}

// Hook to get NEAR authentication challenge
export const useNEARChallenge = (): UseMutationResult<
  NEARChallengeResponse,
  unknown,
  NEARChallengeRequest
> => {
  const getChallengeUrl = `/api/v1/near-challenge`;

  const getChallengeFunction = async (
    data: NEARChallengeRequest
  ): Promise<NEARChallengeResponse> => {
    const response = await api.post<NEARChallengeResponse>(getChallengeUrl, data);
    return response.data;
  };

  return useMutation({
    mutationKey: ["nearChallenge"],
    mutationFn: getChallengeFunction,
  });
};

// Hook to authenticate with NEAR signature
export const useNEARSignatureLogin = (): UseMutationResult<
  NEARLoginResponse,
  unknown,
  NEARSignatureLogin
> => {
  const signatureLoginUrl = `/api/v1/near-auth`;

  const signatureLoginFunction = async (
    data: NEARSignatureLogin
  ): Promise<NEARLoginResponse> => {
    const response = await api.post<NEARLoginResponse>(signatureLoginUrl, data);
    return response.data;
  };

  return useMutation({
    mutationKey: ["nearSignatureLogin"],
    mutationFn: signatureLoginFunction,
  });
};

// Hook to check if NEAR authentication is enabled
export const useNEARAuthEnabled = (): UseMutationResult<
  { enabled: boolean; pool_contract: string; min_stake_amount: string; dev_mode: boolean; superuser: string },
  unknown,
  void
> => {
  const checkEnabledUrl = `/api/v1/near-auth-enabled`;

  const checkEnabledFunction = async () => {
    const response = await api.get(checkEnabledUrl);
    return response.data;
  };

  return useMutation({
    mutationKey: ["nearAuthEnabled"],
    mutationFn: checkEnabledFunction,
  });
};
