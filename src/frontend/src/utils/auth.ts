import { config } from '@/config';

const { AUTH_NEAR_URL,
  SIGN_IN_RESTORE_URL_KEY,
  SIGN_IN_NONCE_KEY,
  SIGN_IN_CALLBACK_PATH,
 } = config;

export function clearSignInNonce() {
  return localStorage.removeItem(SIGN_IN_NONCE_KEY);
}

export function returnUrlToRestoreAfterSignIn() {
  const url = localStorage.getItem(SIGN_IN_RESTORE_URL_KEY) || '/';
  if (url.startsWith(SIGN_IN_CALLBACK_PATH)) return '/';
  return url;
}

export function createAuthUrl(
  message: string,
  recipient: string,
  nonce: string,
  callbackUrl: string = "http://localhost:42205/capture",
) {
  const urlParams = new URLSearchParams({ message, recipient, nonce, callbackUrl });
  return `${AUTH_NEAR_URL}/?${urlParams.toString()}`;
}

export function generateNonce() {
  const nonce = Date.now().toString();
  return nonce.padStart(32, '0');
}