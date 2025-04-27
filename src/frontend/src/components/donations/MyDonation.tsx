import { useEffect, useState } from "react";
import { utils } from "near-api-js";
import { useWalletSelector } from "@near-wallet-selector/react-hook";
import { DonationNearContract } from "@/config";

interface Donation {
    account_id: string;
    total_amount: string;
}

const MyDonation = ({ myDonation }) => {
  const { signedAccountId, viewFunction } = useWalletSelector();
  const [donation, setDonation] = useState<number>(0);
  
  // Update donation amount when `myDonation` changes
  useEffect(() => {
    if (!myDonation) return;

    setDonation(Math.round((Number(donation) + Number(myDonation)) * 100) / 100);
  }, [myDonation]);

  useEffect(() => {
    if (!signedAccountId) return;
  
    const getMyDonations = async () => {
      if (signedAccountId.trim() === "") return;
      console.log("Getting donations for account: ", signedAccountId);
  
      try {
        // Fetch donations for the given account
        const loadedDonation = await viewFunction({
          contractId: DonationNearContract,
          method: "get_donation_for_account",
          args: {
            account_id: signedAccountId,
          },
        });
  
        console.log("Loaded donation data: ", loadedDonation); // Log the response
  
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
    };
  
    getMyDonations();
  }, [signedAccountId]);
  
  

  return (
    <>
        {signedAccountId ? (
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
