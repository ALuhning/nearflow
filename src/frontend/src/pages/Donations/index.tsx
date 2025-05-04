import { useState } from "react";
import DonationBox from "../../components/donations/DonationBox";
import DonationsTable from "../../components/donations/DonationsTable";
import MyDonation from "../../components/donations/MyDonation";
import DonationProgress from "@/components/donations/DonationProgress";
import { useDonationStore } from "@/stores/donationStore";
import { useShallow } from "zustand/react/shallow";

export default function DonationsHome() {
  const [myDonation, setMyDonation] = useState(0);

  const totalRaised = useDonationStore(useShallow((state) => state.totalRaised));
  const topSupporters = useDonationStore(useShallow((state) => state.topSupporters));

  const monthlyGoal = 1000;
  const progress = Math.min((totalRaised / monthlyGoal) * 100, 100);

  return (
    <div className="px-6 py-8 max-w-[90rem] mx-auto">
      {/* Two-column layout */}
      <div className="flex flex-col lg:flex-row gap-10">
        {/* Left Column */}
        <div className="lg:w-3/5 w-full">
          {/* Banner */}
          <div className="mb-6 bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-300 dark:border-yellow-700 p-4 rounded text-sm text-gray-800 dark:text-yellow-100">
            <p>
              <strong className="text-black dark:text-white">Nearflow</strong> is proudly built and maintained by a solo developer at <strong>Vital Point</strong>. No teams. No funding. Just dedication.
              <br />
              <span className="block mt-2">
                If you find Nearflow useful or want to see it grow, consider supporting its future. Every Ⓝ donated helps keep the project alive, maintained, and evolving—with new features, better stability, and long-term vision.
              </span>
            </p>
          </div>


          <h4 className="text-xl font-semibold mb-4">My Donation</h4>
          <MyDonation myDonation={myDonation} />

          <h4 className="text-xl font-semibold mt-6 mb-4">Latest Donations</h4>
          <DonationsTable />
        </div>

        {/* Right Column */}
        <div className="lg:w-2/5 w-full flex flex-col items-center gap-6 min-w-[340px]">
          {/* Donation and Progress */}
          <div className="flex items-center gap-4 w-full">
            <DonationBox setMyDonation={setMyDonation} />
            <div className="flex flex-col items-center">
              <DonationProgress progress={progress} />
              <p className="mt-2 text-sm text-green-700 dark:text-green-400 text-center">
                🎯 Raised: Ⓝ{totalRaised.toFixed(2)} / Ⓝ{monthlyGoal}
              </p>
            </div>
          </div>

          {/* Top Supporters */}
          <div className="w-full bg-white dark:bg-zinc-900 shadow rounded p-4 border dark:border-zinc-700">
            <p className="text-md font-semibold text-center">🏅 Top Supporters</p>
            <ul className="mt-2 list-disc list-inside text-sm text-center">
              {topSupporters.map((s) => (
                <li key={s.account_id}>
                  {s.account_id}: Ⓝ{s.total_donated.toFixed(2)}
                </li>
              ))}
              {topSupporters.length === 0 && (
                <li className="text-zinc-500">Be the first to donate!</li>
              )}
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
