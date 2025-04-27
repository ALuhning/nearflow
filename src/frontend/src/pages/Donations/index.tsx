import DonationBox from "../../components/donations/DonationBox";
import DonationsTable from "../../components/donations/DonationsTable";
import MyDonation from "../../components/donations/MyDonation";
import { useState } from "react";

export default function DonationsHome() {
  const [myDonation, setMyDonation] = useState(0);
  return (
    <div className="p-4 p-sm-5">
      <div className="row">
        <div className="col-sm-8 pe-2 pe-sm-5">
          <h4>My Donation</h4>
          <MyDonation myDonation={myDonation} />
          <h4>Latest Donations</h4>
          <DonationsTable />
        </div>
        <div className="col-sm-4">
          <DonationBox setMyDonation={setMyDonation} />
        </div>
      </div>
    </div>
  );
}