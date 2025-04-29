import { useState } from "react";
import DonationBox from "../../components/donations/DonationBox";
import DonationsTable from "../../components/donations/DonationsTable";
import MyDonation from "../../components/donations/MyDonation";

export default function DonationsHome() {
  const [myDonation, setMyDonation] = useState(0);

  return (
    <div className="p-4 sm:p-5">
        <div className="flex flex-wrap">
            <div className="sm:w-2/3 pr-2 sm:pr-5 mb-4 sm:mb-0">
                <h4 className="text-xl font-semibold mb-4">My Donation</h4>
                <MyDonation myDonation={myDonation} />
                <h4 className="text-xl font-semibold mt-6 mb-4">Latest Donations</h4>
                <DonationsTable />
            </div>
            <div className="sm:w-1/3">
                <DonationBox setMyDonation={setMyDonation} />
            </div>
        </div>
    </div>
  );
}
