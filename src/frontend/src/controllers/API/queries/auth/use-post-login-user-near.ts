import { useMutation, UseMutationResult, useQuery, UseQueryResult } from "@tanstack/react-query";
import { api } from "../../api";
import { getURL } from "../../helpers/constants";

export interface SignAuthMessageInput {
  accountId: string;
  publicKey: string;
  signature: string;
  message: string;
  nonce: string;
  recipient: string;
  callbackUrl: string;
}

export interface SignAuthMessageResponse {
  ok: boolean;
  accountId: string;
  signature: string;
  callbackUrl: string;
}

export interface SessionResponse {
  account_id: string;
  public_key: string;
  signature: string;
  message: string;
  nonce: string;
  recipient: string;
  callback_url: string;
  on_behalf_of: string;
}

export const useSignAuthUrlMessageMutation = (): UseMutationResult<
  SignAuthMessageResponse,
  unknown,
  SignAuthMessageInput
> => {

  const signMessageFn = async (
    payload: SignAuthMessageInput
  ): Promise<SignAuthMessageResponse> => {
    const response = await api.post<SignAuthMessageResponse>(
      `${getURL("AUTH")}/sign-message`,
      payload,
      {
        withCredentials: true,
      }
    );
    return response.data;
  };

  return useMutation({
    mutationKey: ["signAuthUrlMessage"],
    mutationFn: signMessageFn,
  });
};

export const useSessionQuery = (): UseQueryResult<SessionResponse> => {
  const fetchSession = async (): Promise<SessionResponse> => {
    const response = await api.get<SessionResponse>(
      `${getURL("AUTH")}/session`,
      {
        withCredentials: true,
      }
    );
    return response.data;
  };

  return useQuery({
    queryKey: ["authSession"],
    queryFn: fetchSession,
    retry: false,
    refetchOnWindowFocus: false,
  });
};

export const useSignOutMutation = () => {
  return useMutation({
    mutationFn: async (): Promise<{ ok: boolean }> => {
      const response = await api.post(`${getURL("AUTH")}/sign-out`, null, {
        withCredentials: true,
      });
      return response.data;
    },
  });
};