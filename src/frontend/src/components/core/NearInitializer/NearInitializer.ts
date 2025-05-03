import { useNearInitializer } from "../../../hooks/near/near";
import { useWalletInitializer } from "../../../hooks/near/wallet";

export const NearInitializer = () => {
  useNearInitializer();
  useWalletInitializer();
  return null;
};

export default NearInitializer;
