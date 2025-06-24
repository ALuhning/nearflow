/*
Example frontend integration for NEAR account authentication.
This shows how to use the new NEAR authentication endpoints from a web frontend.
*/

// Example JavaScript/TypeScript code for frontend integration

class NEARAuth {
    constructor(baseURL = 'http://localhost:7860') {
        this.baseURL = baseURL;
    }

    /**
     * Check if NEAR authentication is enabled on the server
     */
    async isNearAuthEnabled() {
        try {
            const response = await fetch(`${this.baseURL}/api/v1/login/near-auth-enabled`);
            const data = await response.json();
            return data;
        } catch (error) {
            console.error('Failed to check NEAR auth status:', error);
            return { enabled: false };
        }
    }

    /**
     * Login with NEAR account ID
     * @param {string} nearAccountId - NEAR account ID (e.g., 'user.near')
     * @returns {Promise<Object>} Login response with tokens and user info
     */
    async loginWithNearAccount(nearAccountId) {
        try {
            const response = await fetch(`${this.baseURL}/api/v1/login/near-login`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    near_account_id: nearAccountId
                }),
                credentials: 'include' // Include cookies
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Login failed');
            }

            const data = await response.json();
            return data;
        } catch (error) {
            console.error('NEAR login failed:', error);
            throw error;
        }
    }

    /**
     * Validate NEAR account ID format
     * @param {string} accountId - Account ID to validate
     * @returns {boolean} True if valid format
     */
    isValidNearAccountId(accountId) {
        // Basic validation for NEAR account ID
        const nearPattern = /^[a-z0-9._-]+\.near$|^[a-f0-9]{64}$/;
        return nearPattern.test(accountId);
    }
}

// Example React component using the NEAR authentication
const NEARLoginForm = () => {
    const [nearAccountId, setNearAccountId] = React.useState('');
    const [loading, setLoading] = React.useState(false);
    const [error, setError] = React.useState('');
    const [authEnabled, setAuthEnabled] = React.useState(false);
    
    const nearAuth = new NEARAuth();

    React.useEffect(() => {
        // Check if NEAR auth is enabled
        nearAuth.isNearAuthEnabled().then(data => {
            setAuthEnabled(data.enabled);
        });
    }, []);

    const handleLogin = async (e) => {
        e.preventDefault();
        
        if (!nearAuth.isValidNearAccountId(nearAccountId)) {
            setError('Please enter a valid NEAR account ID (e.g., user.near)');
            return;
        }

        setLoading(true);
        setError('');

        try {
            const result = await nearAuth.loginWithNearAccount(nearAccountId);
            
            // Success! Handle the login response
            console.log('Login successful:', result);
            
            if (result.user_created) {
                alert(`Welcome! Your account has been created. You have ${result.stake_amount} NEAR staked.`);
            } else {
                alert(`Welcome back! You have ${result.stake_amount} NEAR staked.`);
            }
            
            // Redirect or update app state as needed
            window.location.href = '/dashboard';
            
        } catch (error) {
            setError(error.message);
        } finally {
            setLoading(false);
        }
    };

    if (!authEnabled) {
        return <div>NEAR authentication is not enabled on this server.</div>;
    }

    return (
        <div className="near-login-form">
            <h2>Login with NEAR Account</h2>
            <form onSubmit={handleLogin}>
                <div className="form-group">
                    <label htmlFor="near-account">NEAR Account ID:</label>
                    <input
                        id="near-account"
                        type="text"
                        value={nearAccountId}
                        onChange={(e) => setNearAccountId(e.target.value)}
                        placeholder="yourname.near"
                        disabled={loading}
                        required
                    />
                </div>
                
                {error && (
                    <div className="error-message" style={{color: 'red'}}>
                        {error}
                    </div>
                )}
                
                <button 
                    type="submit" 
                    disabled={loading || !nearAccountId}
                    className="login-button"
                >
                    {loading ? 'Verifying...' : 'Login with NEAR'}
                </button>
            </form>
            
            <div className="info-text">
                <p>
                    You must be a staker in the vitalpoint.pool.near contract 
                    with at least 25 NEAR to access this service.
                </p>
            </div>
        </div>
    );
};

// Example usage
export { NEARAuth, NEARLoginForm };
