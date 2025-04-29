import { utils } from "near-api-js";
import { useEffect, useState } from "react";
import { useWalletSelector } from "@near-wallet-selector/react-hook";
import { DonationNearContract } from "@/config";

interface Donation {
    account_id: string;
    total_amount: string;
  }

const DonationsTable = () => {
  const { signedAccountId, viewFunction } = useWalletSelector();
  const [donations, setDonations] = useState<Donation[]>([]);
  const [currentPage, setCurrentPage] = useState(1);
  const [lastPage, setLastPage] = useState(0);
  const donationsPerPage = 5;
  
  const getDonations = async (page: number) => {
    try {
      // Fetch total number of donors
      const number_of_donors = await viewFunction({
        contractId: DonationNearContract, // Replace with your contract ID
        method: "number_of_donors",
      }) as number;
  
      if (number_of_donors === undefined) {
        console.error("Error fetching number of donors.");
        return [];
      }
  
      setLastPage(Math.ceil(number_of_donors / donationsPerPage));
  
      // Calculate fromIndex for pagination
      const fromIndex = (page - 1) * donationsPerPage;
  
      // Fetch donations using pagination
      const donations = await viewFunction({
        contractId: DonationNearContract, // Replace with your contract ID
        method: "get_donations",
        args: {
          from_index: fromIndex,
          limit: donationsPerPage,
        },
      });
  
      if (!donations) {
        console.log("No donations found.");
        return [];
      }
  
      return donations;
    } catch (error) {
      console.error("Error fetching donations:", error);
      return []; // Return an empty array in case of error
    }
  };
  

  useEffect(() => {
    if (!signedAccountId) return;
    getDonations(currentPage).then((loadedDonations) =>
      setDonations(loadedDonations as Donation[]),
    );
  }, [signedAccountId, currentPage]);

  const goToNextPage = () => {
    setCurrentPage((prevPage) => prevPage + 1);
  };

  const goToPrevPage = () => {
    setCurrentPage((prevPage) => prevPage - 1);
  };

  return (
    <div>
        <table className="min-w-full table-auto border-collapse">
            <thead>
            <tr>
                <th className="px-4 py-2 border-b text-left">User</th>
                <th className="px-4 py-2 border-b text-left">Total Donated Ⓝ</th>
            </tr>
            </thead>
            <tbody>
            {donations.map((donation) => (
                <tr key={donation.account_id} className="odd:bg-gray-100">
                <td className="px-4 py-2 border-b">{donation.account_id}</td>
                <td className="px-4 py-2 border-b">{utils.format.formatNearAmount(donation.total_amount)}</td>
                </tr>
            ))}
            </tbody>
        </table>

        <div className="flex items-center mt-4">
            <button
            className={`px-4 py-2 text-white bg-[#1A1D2E] rounded-md text-sm hover:bg-[#141828] focus:outline-none ${currentPage === 1 ? 'opacity-50 cursor-not-allowed' : ''}`}
            onClick={goToPrevPage}
            disabled={currentPage === 1}
            >
            Previous
            </button>
            <span className="mx-4 text-sm">Page {currentPage}</span>
            <button
            className={`px-4 py-2 text-white bg-[#1A1D2E] rounded-md text-sm hover:bg-[#141828] focus:outline-none ${lastPage <= currentPage ? 'opacity-50 cursor-not-allowed' : ''}`}
            onClick={goToNextPage}
            disabled={lastPage <= currentPage}
            >
            Next
            </button>
        </div>
    </div>
  );
};

export default DonationsTable;
