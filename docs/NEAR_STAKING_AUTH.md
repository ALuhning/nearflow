# NEAR Blockchain Staking Authentication

This document describes how to configure and use NEAR blockchain staking verification for user authentication in Nearflow.

## Overview

The NEAR staking authentication feature adds an additional layer of authorization to the standard username/password authentication. When enabled, users must not only provide valid credentials but also be stakers in a specified NEAR staking pool contract with a minimum stake amount.

## Features

- **Blockchain Verification**: Verifies user stake in a NEAR staking pool contract
- **Minimum Stake Requirement**: Configurable minimum stake amount required for access
- **Flexible Configuration**: Can be enabled/disabled via environment variables
- **Error Handling**: Graceful handling of blockchain network issues
- **Logging**: Comprehensive logging for debugging and monitoring

## Configuration

### Environment Variables

Add the following environment variables to your `.env` file:

```bash
# Enable NEAR blockchain staking verification for login authorization
LANGFLOW_ENABLE_NEAR_STAKING_VERIFICATION=true

# NEAR RPC endpoint URL (default: https://rpc.mainnet.near.org)
LANGFLOW_NEAR_RPC_URL=https://rpc.mainnet.near.org

# NEAR staking pool contract address
LANGFLOW_NEAR_POOL_CONTRACT=vitalpoint.pool.near

# Minimum stake amount required in NEAR tokens
LANGFLOW_NEAR_MIN_STAKE_AMOUNT=100

# Disable auto-login when using staking verification
LANGFLOW_AUTO_LOGIN=false
```

### Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `LANGFLOW_ENABLE_NEAR_STAKING_VERIFICATION` | `false` | Enable/disable NEAR staking verification |
| `LANGFLOW_NEAR_RPC_URL` | `https://rpc.mainnet.near.org` | NEAR RPC endpoint URL |
| `LANGFLOW_NEAR_POOL_CONTRACT` | `vitalpoint.pool.near` | Staking pool contract address |
| `LANGFLOW_NEAR_MIN_STAKE_AMOUNT` | `100` | Minimum stake amount in NEAR tokens |

## How It Works

1. **Standard Authentication**: User provides username and password
2. **Username Mapping**: The username is expected to be a NEAR account ID (e.g., `user.near`)
3. **Blockchain Query**: System queries the staking pool contract for user's staking information
4. **Stake Verification**: Checks if user has sufficient stake in the pool
5. **Access Decision**: Grants or denies access based on verification results

## Usage

### User Login Process

When NEAR staking verification is enabled:

1. User enters their NEAR account ID as username (e.g., `alice.near`)
2. User enters their password
3. System performs standard authentication
4. System queries the NEAR blockchain to verify staking status
5. Access is granted only if user meets staking requirements

### Error Messages

Users may see the following error messages:

- **Not a Staker**: "Access denied: You must be a staker in vitalpoint.pool.near to access this service"
- **Insufficient Stake**: "Access denied: Minimum stake of 100 NEAR required. Your current stake: 50 NEAR"
- **Service Unavailable**: "Authentication service temporarily unavailable. Please try again later."

## Implementation Details

### Architecture

The implementation consists of:

- **NEARStakingVerifier**: Core class for blockchain interactions
- **Enhanced Authentication**: Modified auth flow with staking checks
- **Configuration Integration**: Settings management for NEAR-specific config

### Key Components

#### NEARStakingVerifier

Located in `src/backend/base/langflow/services/near/staking.py`:

```python
class NEARStakingVerifier:
    async def verify_staker(self, account_id: str) -> Dict[str, Any]:
        # Verifies staking status and amount
```

#### Enhanced Authentication

Located in `src/backend/base/langflow/services/auth/utils.py`:

```python
async def authenticate_user_with_near_staking(username: str, password: str, db: AsyncSession) -> User | None:
    # Enhanced auth with NEAR staking verification
```

### Database Schema

No database schema changes are required. The feature uses existing user authentication tables and queries the NEAR blockchain directly.

## Security Considerations

- **Blockchain Dependency**: Authentication becomes dependent on NEAR RPC availability
- **Network Security**: Uses HTTPS for all blockchain communications
- **Error Handling**: Fails securely when blockchain is unavailable
- **Logging**: Does not log sensitive information like private keys

## Performance Considerations

- **Network Latency**: Each login requires a blockchain query (typically 100-500ms)
- **Caching**: No caching implemented - each login verifies current stake
- **Rate Limiting**: Subject to NEAR RPC rate limits
- **Timeout Handling**: Built-in timeout for blockchain queries

## Troubleshooting

### Common Issues

1. **Network Connectivity**: Ensure server can reach NEAR RPC endpoint
2. **Invalid Account**: Verify NEAR account ID format (e.g., `user.near`)
3. **Pool Contract**: Ensure specified pool contract exists and is accessible
4. **Minimum Stake**: Check if user's stake meets minimum requirement

### Debugging

Enable debug logging to see detailed staking verification:

```bash
LANGFLOW_LOG_LEVEL=DEBUG
```

### Testing

Use testnet for development:

```bash
LANGFLOW_NEAR_RPC_URL=https://rpc.testnet.near.org
LANGFLOW_NEAR_POOL_CONTRACT=test.pool.testnet
```

## Development

### Running Tests

```bash
# Unit tests
pytest src/backend/tests/unit/services/test_near_staking.py

# Integration tests
pytest src/backend/tests/integration/test_near_staking_auth.py
```

### Local Development

1. Copy the example environment file:
   ```bash
   cp .env.near-staking-example .env
   ```

2. Update the configuration values as needed

3. Start the development server:
   ```bash
   make backend
   ```

## Future Enhancements

- **Caching**: Implement stake amount caching with TTL
- **Multiple Pools**: Support for multiple staking pool contracts
- **Delegation**: Support for delegated staking verification
- **Metrics**: Add monitoring metrics for staking verification
- **Account Mapping**: Map usernames to NEAR account IDs via database
