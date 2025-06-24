/**
 * Frontend Example: NEAR Signature-Based Authentication
 * 
 * This example demonstrates the proper NEAR authentication flow using challenge/response
 * signature verification as recommended by NEAR's official documentation.
 */

class NEARSignatureAuth {
    constructor(backendUrl = 'http://localhost:7860') {
        this.backendUrl = backendUrl;
        this.wallet = null;
        this.selector = null;
    }

    /**
     * Initialize NEAR wallet selector
     * Make sure to include @near-wallet-selector/* packages in your project
     */
    async initWalletSelector() {
        const { setupWalletSelector } = await import("@near-wallet-selector/core");
        const { setupMyNearWallet } = await import("@near-wallet-selector/my-near-wallet");
        const { setupMeteorWallet } = await import("@near-wallet-selector/meteor-wallet");
        const { setupHereWallet } = await import("@near-wallet-selector/here-wallet");

        this.selector = await setupWalletSelector({
            network: "testnet", // or "mainnet"
            modules: [
                setupMyNearWallet(),
                setupMeteorWallet(),
                setupHereWallet(),
            ],
        });

        return this.selector;
    }

    /**
     * Connect to a NEAR wallet
     */
    async connectWallet() {
        if (!this.selector) {
            await this.initWalletSelector();
        }

        const modal = setupModal(this.selector, {
            contractId: "your-contract.testnet" // Optional
        });

        modal.show();
        
        // Wait for wallet connection
        return new Promise((resolve, reject) => {
            const subscription = this.selector.store.observable.subscribe(state => {
                if (state.accounts.length > 0) {
                    this.wallet = this.selector.wallet();
                    subscription.unsubscribe();
                    resolve(state.accounts[0]);
                }
            });
        });
    }

    /**
     * Step 1: Get authentication challenge from backend
     */
    async getChallenge(nearAccountId) {
        try {
            const response = await fetch(`${this.backendUrl}/api/v1/login/near-challenge`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    near_account_id: nearAccountId
                })
            });

            if (!response.ok) {
                throw new Error(`Failed to get challenge: ${response.statusText}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Error getting NEAR challenge:', error);
            throw error;
        }
    }

    /**
     * Step 2: Sign the challenge with NEAR wallet
     */
    async signChallenge(challenge, message, recipient, callbackUrl = null) {
        if (!this.wallet) {
            throw new Error('Wallet not connected');
        }

        try {
            // Convert base64 challenge to Uint8Array
            const challengeBytes = Uint8Array.from(atob(challenge), c => c.charCodeAt(0));

            // Sign the message with the wallet
            const signedMessage = await this.wallet.signMessage({
                message: message,
                recipient: recipient,
                nonce: challengeBytes,
                callbackUrl: callbackUrl
            });

            return signedMessage;
        } catch (error) {
            console.error('Error signing challenge:', error);
            throw error;
        }
    }

    /**
     * Step 3: Authenticate with backend using signature
     */
    async authenticateWithSignature(nearAccountId, signedMessage, challenge, message, recipient) {
        try {
            const response = await fetch(`${this.backendUrl}/api/v1/login/near-auth`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    near_account_id: nearAccountId,
                    public_key: signedMessage.publicKey,
                    signature: signedMessage.signature,
                    challenge: challenge,
                    message: message,
                    recipient: recipient
                }),
                credentials: 'include' // Include cookies
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(`Authentication failed: ${errorData.detail || response.statusText}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Error authenticating with signature:', error);
            throw error;
        }
    }

    /**
     * Complete NEAR authentication flow
     */
    async login() {
        try {
            // Step 1: Connect wallet if not connected
            if (!this.wallet) {
                await this.connectWallet();
            }

            // Get account ID
            const accounts = await this.wallet.getAccounts();
            if (accounts.length === 0) {
                throw new Error('No accounts connected');
            }
            const nearAccountId = accounts[0].accountId;

            // Step 2: Get challenge from backend
            console.log('Getting challenge for account:', nearAccountId);
            const challengeData = await this.getChallenge(nearAccountId);

            // Step 3: Sign the challenge
            console.log('Signing challenge...');
            const signedMessage = await this.signChallenge(
                challengeData.challenge,
                challengeData.message,
                challengeData.recipient
            );

            // Step 4: Authenticate with backend
            console.log('Authenticating with backend...');
            const authResult = await this.authenticateWithSignature(
                nearAccountId,
                signedMessage,
                challengeData.challenge,
                challengeData.message,
                challengeData.recipient
            );

            console.log('Authentication successful:', authResult);
            return authResult;

        } catch (error) {
            console.error('NEAR authentication failed:', error);
            throw error;
        }
    }

    /**
     * Check if NEAR authentication is enabled on the backend
     */
    async checkNearAuthEnabled() {
        try {
            const response = await fetch(`${this.backendUrl}/api/v1/login/near-auth-enabled`);
            if (!response.ok) {
                throw new Error(`Failed to check NEAR auth status: ${response.statusText}`);
            }
            return await response.json();
        } catch (error) {
            console.error('Error checking NEAR auth status:', error);
            throw error;
        }
    }

    /**
     * Logout (disconnect wallet)
     */
    async logout() {
        if (this.wallet) {
            await this.wallet.signOut();
            this.wallet = null;
        }
    }
}

// Usage Example
async function example() {
    const nearAuth = new NEARSignatureAuth('http://localhost:7860');

    try {
        // Check if NEAR auth is enabled
        const authStatus = await nearAuth.checkNearAuthEnabled();
        console.log('NEAR auth status:', authStatus);

        if (!authStatus.enabled) {
            console.log('NEAR authentication is not enabled on the backend');
            return;
        }

        // Perform login
        const result = await nearAuth.login();
        console.log('Login successful!', {
            userCreated: result.user_created,
            stakeAmount: result.stake_amount,
            hasAccessToken: !!result.access_token
        });

        // Now you can make authenticated requests using the access token
        // or rely on the cookies that were set

    } catch (error) {
        console.error('Login failed:', error);
    }
}

// Export for use in modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { NEARSignatureAuth };
}

// Example button click handler for HTML
function handleNearLogin() {
    const nearAuth = new NEARSignatureAuth();
    nearAuth.login()
        .then(result => {
            document.getElementById('login-status').innerHTML = 
                `<div class="success">Login successful! User created: ${result.user_created}</div>`;
        })
        .catch(error => {
            document.getElementById('login-status').innerHTML = 
                `<div class="error">Login failed: ${error.message}</div>`;
        });
}

/* 
HTML Example:

<!DOCTYPE html>
<html>
<head>
    <title>NEAR Authentication Example</title>
    <script src="https://cdn.jsdelivr.net/npm/@near-wallet-selector/core@latest/lib/umd/core.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@near-wallet-selector/my-near-wallet@latest/lib/umd/my-near-wallet.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@near-wallet-selector/meteor-wallet@latest/lib/umd/meteor-wallet.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@near-wallet-selector/here-wallet@latest/lib/umd/here-wallet.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@near-wallet-selector/modal-ui@latest/lib/umd/modal-ui.js"></script>
</head>
<body>
    <h1>NEAR Authentication</h1>
    <button onclick="handleNearLogin()">Login with NEAR</button>
    <div id="login-status"></div>
    <script src="frontend_near_signature_auth.js"></script>
</body>
</html>
*/
