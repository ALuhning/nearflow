# NEAR Authentication Frontend Integration

## Overview
The frontend now includes a complete NEAR authentication system integrated into the login page. Here's what has been implemented:

## Components Created

### 1. NEAR Authentication Hooks (`use-near-auth.ts`)
- `useNEARChallenge()` - Gets authentication challenge from backend
- `useNEARSignatureLogin()` - Submits signed challenge for login
- `useNEARAuthEnabled()` - Checks if NEAR auth is enabled

### 2. NEAR Authentication Component (`NEARAuthComponent`)
- Integrates with NEAR Wallet Selector
- Supports multiple NEAR wallets (MyNearWallet, HERE, Meteor, etc.)
- Handles complete challenge/response flow
- Shows staking requirements to users
- Provides user feedback and error handling

### 3. Updated Login Page
- Adds NEAR authentication option alongside traditional login
- Shows "Or continue with" separator
- Displays staking requirements
- Integrates seamlessly with existing login flow

## Authentication Flow

1. **User clicks "Sign in with NEAR Wallet"**
2. **Wallet Selection Modal appears** (powered by NEAR Wallet Selector)
3. **User selects and connects their NEAR wallet**
4. **Frontend requests challenge** from backend `/api/v1/login/near-challenge`
5. **User signs challenge** using their NEAR wallet
6. **Frontend submits signature** to backend `/api/v1/login/near-auth`
7. **Backend verifies signature AND staking** requirements
8. **User is authenticated** and receives access tokens

## Security Features

✅ **Challenge/Response Flow**: Uses unique nonces to prevent replay attacks
✅ **Signature Verification**: Cryptographically proves account ownership  
✅ **Staking Verification**: Ensures user meets minimum stake requirements
✅ **Multi-Wallet Support**: Works with all major NEAR wallets
✅ **Error Handling**: Provides clear feedback for various failure scenarios

## Configuration Display

The component automatically displays:
- Required minimum stake amount
- Pool contract address  
- Whether NEAR auth is enabled

## User Experience

- **Seamless Integration**: NEAR auth appears naturally in login flow
- **Clear Requirements**: Users see staking requirements upfront
- **Multiple Wallets**: Support for various NEAR wallet options
- **Loading States**: Clear feedback during authentication process
- **Error Messages**: Helpful error messages for troubleshooting

## Required Dependencies

All necessary NEAR packages are already installed:
- `@near-wallet-selector/core` - Core wallet selector
- `@near-wallet-selector/modal-ui` - Modal interface
- Multiple wallet adapters (MyNearWallet, HERE, Meteor, etc.)
- `near-api-js` - NEAR API utilities

## Styling

- Uses existing UI components and styling
- Integrates NEAR Wallet Selector modal styles
- Responsive design matching the rest of the application
- Clear visual separation from traditional login

The implementation provides a complete, secure NEAR authentication system that enforces both signature verification and staking requirements while providing an excellent user experience.
