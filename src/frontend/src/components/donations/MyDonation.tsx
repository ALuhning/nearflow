import { useEffect, useState } from "react";
import { utils } from "near-api-js";
import { useWalletStore } from "@/stores/walletStore";
import { useNearStore } from "@/stores/near";
import { DonationNearContract } from "@/config";
import { useShallow } from "zustand/react/shallow";

interface Donation {
    account_id: string;
    total_amount: string;
}

const MyDonation = ({ myDonation }) => {
  const walletAccount = useWalletStore(useShallow((s) => s.account));
  const wallet = useWalletStore(useShallow((state) => state.wallet));
  const viewAccount = useNearStore.getState().viewAccount;
  const [donation, setDonation] = useState<number>(0);
  
  // Update donation amount when `myDonation` changes
  useEffect(() => {
    if (!myDonation) return;

    setDonation(Math.round((Number(donation) + Number(myDonation)) * 100) / 100);
  }, [myDonation]);

  useEffect(() => {
    const getMyDonations = async () => {
      if (wallet && walletAccount) {
      try {
        // Fetch donations for the given account
        if (!walletAccount || walletAccount.accountId.trim() === "") {
          console.error("Invalid AccountId");
          return;
        }
        
        const loadedDonation = await viewAccount?.viewFunction({
          contractId: DonationNearContract,
          methodName: "get_donation_for_account",
          args: {
            "account_id": walletAccount.accountId,
          },
        }) as Donation;
  
        // Ensure loadedDonation is valid and has total_amount
        if (!loadedDonation || !loadedDonation.total_amount) {
          console.error("Invalid donation data or total_amount not available.");
          setDonation(0); // Set to 0 if no donation data is found or total_amount is missing
          return;
        }
  
        // Format and set the donation amount
        const formattedDonation = utils.format.formatNearAmount(loadedDonation.total_amount) ?? "0";
        setDonation(parseFloat(formattedDonation)); // Convert string to number
      } catch (error) {
        console.error("Error fetching donation data:", error);
        setDonation(0); // Set to 0 in case of error
      }
    }
    };
    
    getMyDonations();
  }, [walletAccount?.accountId]);
  
  
  

  return (
    <>
        {walletAccount?.accountId ? (
            <p className="mb-3 text-gray-700">
            You have donated <strong>{donation} NEAR</strong> to the cause.
            </p>
        ) : (
            <p className="mb-3 text-gray-500">
            Please sign in with your NEAR wallet to make a donation.
            </p>
        )}
        </>
  );
};

export default MyDonation;
