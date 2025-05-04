import { utils } from "near-api-js";
import { useEffect, useState } from "react";
import { useWalletStore } from "@/stores/walletStore";
import { useNearStore } from "@/stores/near";
import { DonationNearContract } from "@/config";
import { useDonationStore } from "@/stores/donationStore";
import { useShallow } from "zustand/react/shallow";

interface Donation {
  account_id: string;
  total_amount: string;
}

const DonationsTable = () => {
  const walletAccount = useWalletStore(useShallow((s) => s.account));
  const viewAccount = useNearStore.getState().viewAccount;

  const setTotalRaised = useDonationStore((state) => state.setTotalRaised);
  const setTopSupporters = useDonationStore((state) => state.setTopSupporters);

  const [donations, setDonations] = useState<Donation[]>([]);
  const [currentPage, setCurrentPage] = useState(1);
  const [lastPage, setLastPage] = useState(0);
  const donationsPerPage = 5;

  const fetchAllDonations = async (numberOfDonors: number): Promise<Donation[]> => {
    const all: Donation[] = [];
    for (let from = 0; from < numberOfDonors; from += donationsPerPage) {
      const batch: Donation[] = await viewAccount?.viewFunction({
        contractId: DonationNearContract,
        methodName: "get_donations",
        args: {
          from_index: from,
          limit: donationsPerPage,
        },
      }) || [];
      all.push(...batch);
    }
    return all;
  };

  const getDonations = async (page: number) => {
    try {
      const number_of_donors = await viewAccount?.viewFunction({
        contractId: DonationNearContract,
        methodName: "number_of_donors",
      }) as number;

      if (!number_of_donors) return [];

      const fromIndex = (page - 1) * donationsPerPage;
      const paginatedDonations: Donation[] = await viewAccount?.viewFunction({
        contractId: DonationNearContract,
        methodName: "get_donations",
        args: {
          from_index: fromIndex,
          limit: donationsPerPage,
        },
      }) || [];

      setLastPage(Math.ceil(number_of_donors / donationsPerPage));

      // Fetch all donations in batches of 5
      const allDonations = await fetchAllDonations(number_of_donors);

      const formattedSupporters = allDonations.map((d) => ({
        account_id: d.account_id,
        total_donated: parseFloat(utils.format.formatNearAmount(d.total_amount)),
      }));

      const totalRaised = formattedSupporters.reduce((sum, s) => sum + s.total_donated, 0);

      setTotalRaised(totalRaised);
      setTopSupporters(allDonations, []);

      return paginatedDonations;
    } catch (error) {
      console.error("Error fetching donations:", error);
      return [];
    }
  };

  useEffect(() => {
    if (!walletAccount?.accountId) return;
    getDonations(currentPage).then((loaded) => setDonations(loaded as Donation[]));
  }, [walletAccount?.accountId, currentPage]);

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
              <td className="px-4 py-2 border-b">
                {utils.format.formatNearAmount(donation.total_amount)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      <div className="flex items-center mt-4">
        <button
          className={`px-4 py-2 text-white bg-[#1A1D2E] rounded-md text-sm hover:bg-[#141828] focus:outline-none ${currentPage === 1 ? "opacity-50 cursor-not-allowed" : ""}`}
          onClick={() => setCurrentPage((p) => p - 1)}
          disabled={currentPage === 1}
        >
          Previous
        </button>
        <span className="mx-4 text-sm">Page {currentPage}</span>
        <button
          className={`px-4 py-2 text-white bg-[#1A1D2E] rounded-md text-sm hover:bg-[#141828] focus:outline-none ${lastPage <= currentPage ? "opacity-50 cursor-not-allowed" : ""}`}
          onClick={() => setCurrentPage((p) => p + 1)}
          disabled={lastPage <= currentPage}
        >
          Next
        </button>
      </div>
    </div>
  );
};

export default DonationsTable;
