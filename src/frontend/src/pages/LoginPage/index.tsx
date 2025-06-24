import LangflowLogo from "@/assets/vitalpoint.svg?react";
import { JSX, useContext, useState, useEffect } from "react";
import { Button } from "../../components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../../components/ui/tabs";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "../../components/ui/accordion";
import SimpleNEARAuth from "../../components/SimpleNEARAuth";
import NEARTestAuth from "../../components/NEARTestAuth";
import { useNEARAuthEnabled } from "../../controllers/API/queries/auth";
import { SIGNIN_ERROR_ALERT } from "../../constants/alerts_constants";
import { AuthContext } from "../../contexts/authContext";
import useAuthStore from "../../stores/authStore";
import useAlertStore from "../../stores/alertStore";
import { useShallow } from "zustand/react/shallow";

export default function LoginPage(): JSX.Element {
  const [nearDevMode, setNearDevMode] = useState<boolean | null>(null);
  const [nearWalletConnected, setNearWalletConnected] = useState<boolean>(false);
  const [nearAccountId, setNearAccountId] = useState<string | null>(null);
  const [stakingMeetsRequirements, setStakingMeetsRequirements] = useState<boolean | null>(null);
  const [checkingWalletAndStaking, setCheckingWalletAndStaking] = useState<boolean>(false);
  const [isSuperuser, setIsSuperuser] = useState<boolean>(false);
  const [superuserAccount, setSuperuserAccount] = useState<string | null>(null);
  const [userExists, setUserExists] = useState<boolean | null>(null);
  const [nearAuthConfig, setNearAuthConfig] = useState<any>(null);
  
  const { login, userData } = useContext(AuthContext);
  const setErrorData = useAlertStore(useShallow((state) => state.setErrorData));
  const isAuthenticated = useAuthStore(useShallow((state) => state.isAuthenticated));
  const isAdmin = useAuthStore(useShallow((state) => state.isAdmin));
  const nearAuthEnabledMutation = useNEARAuthEnabled();

  // Check admin status from auth store (this gets set automatically when user data is loaded)
  useEffect(() => {
    console.log("Admin status from auth store:", isAdmin);
    setIsSuperuser(isAdmin);
  }, [isAdmin]);

  // Check NEAR dev mode from backend
  useEffect(() => {
    nearAuthEnabledMutation.mutate(undefined, {
      onSuccess: (data) => {
        setNearDevMode(data.dev_mode || false);
        setSuperuserAccount(data.superuser || null);
        setNearAuthConfig(data);
      },
      onError: (error) => {
        console.error('Failed to check NEAR dev mode:', error);
        setNearDevMode(false);
      }
    });
  }, []);

  // Check NEAR wallet connection and staking status
  useEffect(() => {
    const checkNearWalletAndStaking = async () => {
      if (isAuthenticated) return; // Skip if already fully authenticated
      
      setCheckingWalletAndStaking(true);
      
      try {
        // Import NEAR wallet selector to check connection
        const { setupWalletSelector } = await import("@near-wallet-selector/core");
        const { setupHereWallet } = await import("@near-wallet-selector/here-wallet");
        const { setupMeteorWallet } = await import("@near-wallet-selector/meteor-wallet");
        
        const walletSelector = await setupWalletSelector({
          network: "mainnet",
          modules: [
            setupHereWallet(),
            setupMeteorWallet(),
          ],
        });
        
        const isSignedIn = walletSelector.isSignedIn();
        if (isSignedIn) {
          const wallet = await walletSelector.wallet();
          const accounts = await wallet.getAccounts();
          
          if (accounts && accounts.length > 0) {
            const accountId = accounts[0].accountId;
            setNearWalletConnected(true);
            setNearAccountId(accountId);
            
            // Check if this account is the designated superuser
            const isAccountSuperuser = accountId === superuserAccount;
            setIsSuperuser(isAccountSuperuser);
            
            // Check if user already exists
            try {
              const userExistsResponse = await fetch(`http://localhost:7860/api/v1/check-user-exists/${accountId}`);
              if (userExistsResponse.ok) {
                const userData = await userExistsResponse.json();
                setUserExists(userData.exists);
              } else {
                setUserExists(false);
              }
            } catch (error) {
              console.error("Error checking user existence:", error);
              setUserExists(false);
            }
            
            // Skip staking check for superusers
            if (isAccountSuperuser) {
              setStakingMeetsRequirements(true); // Superusers always meet requirements
            } else {
              // Check staking for this account
              try {
                const response = await fetch(`http://localhost:7860/api/v1/near-stake-check/${accountId}`);
                if (response.ok) {
                  const data = await response.json();
                  setStakingMeetsRequirements(data.meets_requirements);
                } else {
                  setStakingMeetsRequirements(false);
                }
              } catch (error) {
                console.error("Error checking staking:", error);
                setStakingMeetsRequirements(false);
              }
            }
          } else {
            setNearWalletConnected(false);
            setNearAccountId(null);
            setStakingMeetsRequirements(null);
          }
        } else {
          setNearWalletConnected(false);
          setNearAccountId(null);
          setStakingMeetsRequirements(null);
        }
      } catch (error) {
        console.error("Error checking NEAR wallet:", error);
        setNearWalletConnected(false);
        setNearAccountId(null);
        setStakingMeetsRequirements(null);
      } finally {
        setCheckingWalletAndStaking(false);
      }
    };
    
    checkNearWalletAndStaking();
  }, [isAuthenticated, superuserAccount]);

  // Separate effect to re-check staking when account changes (for account switching)
  useEffect(() => {
    const recheckStaking = async () => {
      if (!nearAccountId || isAuthenticated) return;
      
      setCheckingWalletAndStaking(true);
      setStakingMeetsRequirements(null); // Reset to null while checking
      
      try {
        // Check if this account is the designated superuser  
        const isAccountSuperuser = nearAccountId === superuserAccount;
        setIsSuperuser(isAccountSuperuser);
        
        // Check if user already exists
        try {
          const userExistsResponse = await fetch(`http://localhost:7860/api/v1/check-user-exists/${nearAccountId}`);
          if (userExistsResponse.ok) {
            const userData = await userExistsResponse.json();
            setUserExists(userData.exists);
          } else {
            setUserExists(false);
          }
        } catch (error) {
          console.error("Error checking user existence:", error);
          setUserExists(false);
        }
        
        // Skip staking check for superusers
        if (isAccountSuperuser) {
          setStakingMeetsRequirements(true); // Superusers always meet requirements
        } else {
          const response = await fetch(`http://localhost:7860/api/v1/near-stake-check/${nearAccountId}`);
          if (response.ok) {
            const data = await response.json();
            setStakingMeetsRequirements(data.meets_requirements);
          } else {
            setStakingMeetsRequirements(false);
          }
        }
      } catch (error) {
        console.error("Error rechecking staking:", error);
        setStakingMeetsRequirements(false);
      } finally {
        setCheckingWalletAndStaking(false);
      }
    };
    
    recheckStaking();
  }, [nearAccountId, isAuthenticated, superuserAccount]);

  return (
    <div className="min-h-screen w-full bg-background overflow-auto">
      <div className="container mx-auto px-4 py-8">
        <div className={`${isSuperuser ? 'max-w-2xl mx-auto' : 'grid lg:grid-cols-2 gap-8 max-w-6xl mx-auto'}`}>
          
          {/* Left Column - Authentication */}
          <div className="flex items-start justify-center lg:items-center">
            <div className="w-full max-w-md">
              <div className="text-center lg:text-left mb-6">
                <div className="flex items-center justify-center lg:justify-start gap-3 mb-4">
                  <LangflowLogo className="h-10 w-10 scale-[1.5]" />
                  <div>
                    <h1 className="text-3xl font-bold text-gray-900">Welcome to NearFlow</h1>
                    <p className="text-gray-600">AI Development Platform on NEAR Protocol</p>
                  </div>
                </div>
              </div>
              <div className="bg-white rounded-lg shadow-xl p-8">
                <div className="text-center mb-6">
                  <h2 className="text-2xl font-bold text-gray-900 mb-2">Access NearFlow</h2>
                  <p className="text-gray-600">Connect your NEAR wallet to get started</p>
                </div>

                <Tabs defaultValue="near-auth" className="w-full">
                  <TabsList className="grid w-full grid-cols-1">
                    <TabsTrigger value="near-auth" className="data-[state=active]:bg-blue-500 data-[state=active]:text-white">
                      NEAR Wallet Authentication
                    </TabsTrigger>
                  </TabsList>
                  
                  <TabsContent value="near-auth" className="space-y-4">
                    {isAuthenticated ? (
                      // User is fully authenticated
                      <div className="text-center py-8">
                        <div className="text-green-600 text-xl mb-4">✓</div>
                        <h3 className="font-semibold text-green-800 mb-2">Already Authenticated</h3>
                        <p className="text-green-700 text-sm mb-4">You are already logged in with your NEAR account.</p>
                        <Button
                          onClick={() => window.location.href = "/"}
                          className="w-full"
                        >
                          Go to Dashboard
                        </Button>
                      </div>
                    ) : nearWalletConnected && stakingMeetsRequirements === null && checkingWalletAndStaking ? (
                      // Checking staking status for connected wallet
                      <div className="text-center py-8">
                        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto mb-4"></div>
                        <h3 className="font-semibold text-gray-800 mb-2">Checking Staking Status</h3>
                        <p className="text-gray-600 text-sm">Verifying staking requirements for account <strong>{nearAccountId}</strong>...</p>
                      </div>
                    ) : nearWalletConnected && stakingMeetsRequirements === false && !isSuperuser ? (
                      // User has wallet connected but staking requirements not met (only show for non-superusers)
                      <div className="space-y-4">
                        <div className="text-center bg-yellow-50 p-4 rounded-md border border-yellow-200">
                          <div className="font-medium text-yellow-800 mb-2">🔒 Staking Required</div>
                          <div className="text-yellow-700 text-sm space-y-2">
                            <div>Account <strong>{nearAccountId}</strong> is connected but needs to meet staking requirements.</div>
                            <div>Please stake at least 25 NEAR with <strong>vitalpoint.pool.near</strong> to access NearFlow.</div>
                          </div>
                        </div>
                        
                        {/* Show the SimpleNEARAuth component for staking flow */}
                        {nearDevMode === true ? (
                          <NEARTestAuth 
                            onLoginStart={() => {}}
                            onLoginComplete={() => {}}
                            onLoginError={(error) => {
                              setErrorData({
                                title: SIGNIN_ERROR_ALERT,
                                list: [error],
                              });
                            }}
                          />
                        ) : nearDevMode === false ? (
                          <SimpleNEARAuth 
                            hideStakingRequiredMessage={true}
                            isSuperuser={isSuperuser}
                            userExists={userExists ?? false}
                            nearConfig={nearAuthConfig}
                            onLoginStart={() => {}}
                            onLoginComplete={() => {}}
                            onLoginError={(error) => {
                              setErrorData({
                                title: SIGNIN_ERROR_ALERT,
                                list: [error],
                              });
                            }}
                            onAccountChange={(accountId, userExists, isSuperuser, stakingMeetsRequirements) => {
                              console.log("LoginPage: Account changed:", { accountId, userExists, isSuperuser, stakingMeetsRequirements });
                              if (accountId) {
                                setNearAccountId(accountId);
                                setNearWalletConnected(true);
                              } else {
                                setNearAccountId(null);
                                setNearWalletConnected(false);
                              }
                              setUserExists(userExists);
                              setIsSuperuser(isSuperuser);
                              setStakingMeetsRequirements(stakingMeetsRequirements);
                            }}
                          />
                        ) : (
                          <div className="text-center py-4">
                            <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-500 mx-auto mb-2"></div>
                            <div className="text-xs text-muted-foreground">Loading...</div>
                          </div>
                        )}
                      </div>
                    ) : nearWalletConnected && (stakingMeetsRequirements === true || isSuperuser) ? (
                      // User has wallet connected and meets staking requirements or is a superuser - ready to authenticate
                      <div className="space-y-4">
                        <div className={`text-center p-4 rounded-md border ${isSuperuser ? 'bg-purple-50 border-purple-200' : 'bg-green-50 border-green-200'}`}>
                          <div className={`font-medium mb-2 ${isSuperuser ? 'text-purple-800' : 'text-green-800'}`}>
                            {isSuperuser ? '👑 Admin Access' : '✅ Staking Requirements Met'}
                          </div>
                          <div className={`text-sm space-y-2 ${isSuperuser ? 'text-purple-700' : 'text-green-700'}`}>
                            <div>Account <strong>{nearAccountId}</strong> {isSuperuser ? 'has administrative privileges.' : 'meets staking requirements.'}</div>
                            <div>Click below to {userExists ? 'sign in to' : 'create'} your NearFlow account.</div>
                          </div>
                        </div>
                        
                        {/* Show the SimpleNEARAuth component for final authentication */}
                        {nearDevMode === true ? (
                          <NEARTestAuth 
                            onLoginStart={() => {}}
                            onLoginComplete={() => {}}
                            onLoginError={(error) => {
                              setErrorData({
                                title: SIGNIN_ERROR_ALERT,
                                list: [error],
                              });
                            }}
                          />
                        ) : nearDevMode === false ? (
                          <SimpleNEARAuth 
                            isSuperuser={isSuperuser}
                            userExists={userExists ?? false}
                            nearConfig={nearAuthConfig}
                            onLoginStart={() => {}}
                            onLoginComplete={() => {}}
                            onLoginError={(error) => {
                              setErrorData({
                                title: SIGNIN_ERROR_ALERT,
                                list: [error],
                              });
                            }}
                          />
                        ) : (
                          <div className="text-center py-4">
                            <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-500 mx-auto mb-2"></div>
                            <div className="text-xs text-muted-foreground">Loading...</div>
                          </div>
                        )}
                      </div>
                    ) : (
                      // User needs to connect wallet and authenticate
                      <>
                        <div className="bg-blue-50 rounded-lg p-4 mb-4">
                          <div className="flex items-start space-x-3">
                            <div className="text-blue-500 text-xl">{isSuperuser ? '�' : '�🔐'}</div>
                            <div>
                              <h3 className="font-semibold text-blue-800 mb-1">
                                {isSuperuser ? 'Admin NEAR Authentication' : 'Secure NEAR Authentication'}
                              </h3>
                              <p className="text-blue-700 text-sm">
                                {isSuperuser 
                                  ? 'Connect your NEAR wallet to authenticate with administrative privileges.' 
                                  : 'Connect any NEAR wallet, verify your staking requirements, and create your NearFlow account - all in one seamless flow.'
                                }
                              </p>
                            </div>
                          </div>
                        </div>

                        {/* NEAR Authentication - Conditional based on dev mode */}
                        {nearDevMode === true ? (
                          <NEARTestAuth 
                            onLoginStart={() => {}}
                            onLoginComplete={() => {}}
                            onLoginError={(error) => {
                              setErrorData({
                                title: SIGNIN_ERROR_ALERT,
                                list: [error],
                              });
                            }}
                          />
                        ) : nearDevMode === false ? (
                          <SimpleNEARAuth 
                            isSuperuser={isSuperuser}
                            userExists={userExists ?? false}
                            nearConfig={nearAuthConfig}
                            onLoginStart={() => {}}
                            onLoginComplete={() => {}}
                            onLoginError={(error) => {
                              setErrorData({
                                title: SIGNIN_ERROR_ALERT,
                                list: [error],
                              });
                            }}
                            onAccountChange={(accountId, userExists, isSuperuser, stakingMeetsRequirements) => {
                              console.log("LoginPage: Account changed (third instance):", { accountId, userExists, isSuperuser, stakingMeetsRequirements });
                              if (accountId) {
                                setNearAccountId(accountId);
                                setNearWalletConnected(true);
                              } else {
                                setNearAccountId(null);
                                setNearWalletConnected(false);
                              }
                              setUserExists(userExists);
                              setIsSuperuser(isSuperuser);
                              setStakingMeetsRequirements(stakingMeetsRequirements);
                            }}
                          />
                        ) : (
                          <div className="text-center py-8">
                            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto mb-4"></div>
                            <div className="text-sm text-muted-foreground">
                              Loading NEAR authentication...
                            </div>
                          </div>
                        )}
                      </>
                    )}

                    {!isAuthenticated && !nearWalletConnected && (
                      <div className="mt-6 pt-6 border-t border-gray-200">
                        <div className="text-center space-y-2">
                          <p className="text-sm text-gray-600">New to NEAR Protocol?</p>
                          <Button
                            variant="outline"
                            onClick={() => window.open("https://meteorwallet.app/", "_blank")}
                            className="w-full"
                          >
                            Create NEAR Wallet
                          </Button>
                        </div>
                      </div>
                    )}
                  </TabsContent>
                </Tabs>
              </div>

              {!isAuthenticated && (
                <div className="mt-6 text-center text-sm text-gray-600">
                  {nearWalletConnected && checkingWalletAndStaking && stakingMeetsRequirements === null ? (
                    <p>{isSuperuser ? "Verifying admin access for your connected NEAR wallet..." : "Checking staking requirements for your connected NEAR wallet..."}</p>
                  ) : nearWalletConnected && stakingMeetsRequirements === false && !isSuperuser ? (
                    <p>Complete your staking requirement to access NearFlow. Your NEAR wallet is already connected.</p>
                  ) : nearWalletConnected && (stakingMeetsRequirements === true || isSuperuser) ? (
                    <p>Your NEAR wallet is connected and {isSuperuser ? "admin access is" : "staking requirements are"} verified. Complete authentication above to access NearFlow.</p>
                  ) : (
                    <p>By connecting your wallet, you agree to our terms of service{!isSuperuser ? " and acknowledge that NearFlow requires a minimum stake of 25 NEAR with the Vital Point Validator (vitalpoint.pool.near)" : ""}.</p>
                  )}
                </div>
              )}
            </div>
          </div>

          {/* Right Column - Information about Staking (hidden for superusers) */}
          {!isSuperuser && (
            <div className="space-y-6">
            <div className="bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-lg p-6">
              <h2 className="text-xl font-bold mb-3">Stake to Join the VP Guild</h2>
              <p className="mb-4">A network of innovators, builders, and visionaries shaping the future of decentralized AI and Web3 - powered by NEAR.</p>
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <div className="font-semibold">🚀 Stake, Belong, Build & Create</div>
                </div>
                <div>
                  <div className="font-semibold">💡 Open Innovation</div>
                  <div className="opacity-90">Collaborative development</div>
                </div>
              </div>
            </div>
            <div className="bg-white rounded-lg shadow-lg p-6">
              <Accordion type="single" collapsible className="w-full">
                <AccordionItem value="what-is-staking">
                  <AccordionTrigger className="text-lg font-semibold text-gray-800">
                    💰 What is Staking?
                  </AccordionTrigger>
                  <AccordionContent className="prose prose-sm max-w-none text-gray-700">
                    <div className="space-y-6">
                      <div className="border-l-4 border-blue-500 pl-4">
                        <div className="space-y-3 text-sm">
                          <p>Think of staking like putting money into a high-interest savings account but for crypto. The easiest way to stake is to delegate your NEAR to a trusted validator.</p>
                          
                          <p>When you stake your NEAR tokens, you're locking them into the network temporarily to help keep it secure and running smoothly, like how a bank uses your deposited money to support its operations and loans.</p>
                          
                          <p>In return, the network pays you rewards - like interest - for your contribution. These rewards come in the form of more NEAR tokens, and they're earned and distributed automatically.</p>
                          
                          <p>NEAR staking has historically generated a rate of return (since mainnet launch in 2020) of around 8–10% annually.</p>
                          
                          <p>The Vital Point validator automatically allocates 5% of the rewards to the guild treasury and the remaining ~4% is allocated to members proportionate to their stake.</p>
                        </div>
                        
                        <div className="mt-4 bg-blue-50 rounded-lg p-3">
                          <h4 className="font-semibold text-blue-800 mb-2">🔐 Key Points:</h4>
                          <ul className="text-blue-700 text-sm space-y-1">
                            <li>You stay in control of your funds - they never leave your wallet</li>
                            <li>You can unstake anytime (though there is a short waiting period)</li>
                            <li>There's no risk of losing your funds</li>
                            <li>Your stake makes you a VP guild member and opens the door to guild rewards and support</li>
                          </ul>
                        </div>
                      </div>
                    </div>
                    
                    <div className="bg-blue-50 rounded-lg p-4 mt-6">
                      <h3 className="font-semibold text-blue-800 mb-2">Why 25 NEAR Minimum?</h3>
                      <p className="text-blue-700">The 25 NEAR minimum stake requirement ensures committed community members while supporting the infrastructure costs of running NearFlow and the Vital Point Guild ecosystem.</p>
                    </div>
                  </AccordionContent>
                </AccordionItem>
                
                <AccordionItem value="hardware-wallet-help">
                  <AccordionTrigger className="text-lg font-semibold text-gray-800">
                    🔐 Using Hardware Wallets (Ledger)
                  </AccordionTrigger>
                  <AccordionContent className="prose prose-sm max-w-none text-gray-700">
                    <div className="space-y-4">
                      <div className="bg-orange-50 border-l-4 border-orange-400 p-4 rounded-r-lg">
                        <h4 className="font-semibold text-orange-800 mb-2">📱 Before You Start</h4>
                        <p className="text-orange-700 text-sm">Hardware wallets like Ledger require additional setup steps for NEAR authentication. Follow these steps carefully:</p>
                      </div>
                      
                      <div className="space-y-3">
                        <div className="bg-white border border-gray-200 rounded-lg p-4">
                          <h5 className="font-semibold text-gray-800 mb-2">1. 🔌 Connect Your Ledger</h5>
                          <ul className="text-sm text-gray-700 space-y-1 ml-4">
                            <li>• Connect your Ledger device to your computer via USB</li>
                            <li>• Enter your PIN to unlock the device</li>
                            <li>• Make sure your device is not sleeping/locked</li>
                          </ul>
                        </div>
                        
                        <div className="bg-white border border-gray-200 rounded-lg p-4">
                          <h5 className="font-semibold text-gray-800 mb-2">2. 📱 Open NEAR App</h5>
                          <ul className="text-sm text-gray-700 space-y-1 ml-4">
                            <li>• Navigate to and open the <strong>NEAR app</strong> on your Ledger</li>
                            <li>• The screen should show "NEAR ready" or similar</li>
                            <li>• Ensure you have the latest version of the NEAR app installed</li>
                          </ul>
                        </div>
                        
                        <div className="bg-white border border-gray-200 rounded-lg p-4">
                          <h5 className="font-semibold text-gray-800 mb-2">3. 🌐 Browser Settings</h5>
                          <ul className="text-sm text-gray-700 space-y-1 ml-4">
                            <li>• Use Chrome, Firefox, or Edge (Safari may have issues)</li>
                            <li>• Disable any ad blockers that might interfere</li>
                            <li>• Allow popup windows for wallet connections if prompted</li>
                          </ul>
                        </div>
                      </div>                        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                        <h4 className="font-semibold text-red-800 mb-2">⚠️ Wallet Compatibility</h4>
                        <div className="text-sm text-red-700 space-y-3">
                          <div><strong>✅ Recommended Hardware Wallet Compatible Options:</strong></div>
                          <ul className="ml-4 space-y-1">
                            <li>• <strong>MeteorWallet:</strong> Excellent Ledger support with smooth authentication</li>
                            <li>• <strong>HereWallet:</strong> Good Ledger compatibility and user experience</li>
                          </ul>
                          
                          <div className="mt-3"><strong>❌ Not Supported:</strong></div>
                          <ul className="ml-4 space-y-1">
                            <li>• <strong>MyNearWallet:</strong> Not supported in NearFlow due to Ledger compatibility issues</li>
                            <li>• Known problems with key verification when using Ledger devices</li>
                            <li>• If you have an existing MyNearWallet account, you can import it into MeteorWallet</li>
                          </ul>
                        </div>
                      </div>
                      
                      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                        <h4 className="font-semibold text-blue-800 mb-2">💡 Alternative Options</h4>
                        <p className="text-blue-700 text-sm">If you continue to have issues with your Ledger, you can:</p>
                        <ul className="text-blue-700 text-sm ml-4 mt-2 space-y-1">
                          <li>• Use a compatible hot wallet (MeteorWallet, HereWallet) for authentication</li>
                          <li>• Import your account into a compatible software wallet temporarily</li>
                          <li>• Contact support if you need assistance with wallet setup</li>
                        </ul>
                      </div>
                    </div>
                  </AccordionContent>
                </AccordionItem>
              </Accordion>
            </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
