import { utils } from "near-api-js";
import { useState } from "react";
import { useWalletSelector } from "@near-wallet-selector/react-hook";
import { DonationNearContract } from "@/config";
import axios from "axios";

const DonationForm = ({ setMyDonation }) => {
  const { callFunction } = useWalletSelector();
  const [amount, setAmount] = useState<number>(0);
  
  const setDonation = async (amount) => {
    try {
      console.log("Fetching donation data for amount:", amount);
  
      // Perform axios request
      const response = await axios.get(
        "https://api.coingecko.com/api/v3/simple/price?ids=near&vs_currencies=usd"
      );
  
      // Log the response to check structure
      console.log("Raw data received:", response.data);
  
      // Validate the structure of the received data
      if (!response.data || !response.data["near"] || !response.data["near"]["usd"]) {
        throw new Error("Invalid data structure received from API");
      }
  
      const near2usd = response.data["near"]["usd"];
      const amount_in_near = amount / near2usd;
      const rounded_two_decimals = Math.round(amount_in_near * 100) / 100;
  
      // Update the amount state with the calculated value
      setAmount(rounded_two_decimals);
    } catch (error) {
      if (error instanceof Error) {
        console.error("Error in donation calculation:", error.message);
      } else {
        console.error("Unknown error in donation calculation:", error);
      }
      setAmount(0); // Fallback to 0 on error
    }
  };
  
  const handleSubmit = async (event) => {
    // event.preventDefault();
    let deposit = utils.format.parseNearAmount(amount.toString()) || "0";
    let response = await callFunction({
        contractId: DonationNearContract,
        method: "donate",
        deposit,
      })
    if(response) {
      setMyDonation(amount);
    } else {
      setMyDonation(-Number(amount));
    }
  };
  
  return (
    <>
      <div className="mb-4">
        <div className="grid grid-cols-4 gap-4">
          {[10, 20, 50, 100].map((amount) => (
            <div className="col-span-1" key={amount}>
              <button
                className="w-full px-4 py-2 bg-[#1A1D2E] text-white rounded hover:bg-[#141828]"
                onClick={() => setDonation(amount)}
              >
                $ {amount}
              </button>
            </div>
          ))}
        </div>
      </div>

      
        <div className="mb-4">
          <label htmlFor="donation" className="block text-sm font-medium text-gray-700">
            Donation amount (in Ⓝ)
          </label>
          <div className="flex items-center mt-2">
            <input
              id="donation"
              value={amount}
              type="number"
              min="0"
              step="0.01"
              onChange={(e) => setAmount(parseFloat(e.target.value) || 0)}
              className="w-full px-4 py-2 border border-gray-300 rounded-l-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            <span className="px-4 py-2 bg-gray-200 rounded-r-md">Ⓝ</span>
          </div>
        </div>

        <button
          onClick={handleSubmit}
          className="w-full px-4 py-2 bg-[#1A1D2E] text-white rounded hover:bg-[#141828]"
        >
          Donate
        </button>
     
    </>
  );
};

export default DonationForm;
