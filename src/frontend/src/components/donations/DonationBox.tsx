import DonationForm from "./DonationForm";
import { useWalletStore } from "@/stores/walletStore";
import { useShallow } from "zustand/react/shallow";

const DonationBox = ({ setMyDonation }) => {
  const walletAccount = useWalletStore(useShallow((s) => s.account));

  return (
    <div className="mt-4 rounded-lg border border-gray-200 shadow-lg w-full min-w-[320px]">
      <div className="p-3 text-center bg-gray-100">
        <h4 className="text-xl font-semibold">
          <strong>Donate to</strong>
        </h4>
      </div>
      <div className="bg-gray-50 p-3">
        {walletAccount?.accountId ? (
          <DonationForm setMyDonation={setMyDonation} />
        ) : (
          <p className="mb-3 text-gray-700">
            Please sign in with your NEAR wallet to make a donation.
          </p>
        )}
      </div>
    </div>
  );
};

export default DonationBox;
