import { create } from "zustand";
import { utils } from "near-api-js";

interface Supporter {
  account_id: string;
  total_donated: number; // In NEAR, already formatted
}

interface Donation {
  account_id: string;
  total_amount: string;
}

interface DonationStore {
  refreshDonations: boolean;
  totalRaised: number;
  topSupporters: Supporter[];

  triggerRefresh: () => void;
  resetRefresh: () => void;

  setTotalRaised: (amount: number) => void;

  // Merge external supporters and on-chain donations
  setTopSupporters: (donations: Donation[], externalSupporters: Supporter[]) => void;
}

export const useDonationStore = create<DonationStore>((set) => ({
  refreshDonations: false,
  totalRaised: 0,
  topSupporters: [],

  triggerRefresh: () => set({ refreshDonations: true }),
  resetRefresh: () => set({ refreshDonations: false }),

  setTotalRaised: (amount) => set({ totalRaised: amount }),

  setTopSupporters: (donations, externalSupporters) => {
    // Convert on-chain donations to supporter format
    const onChainSupporters: Record<string, Supporter> = {};

    for (const donation of donations) {
      const parsed = parseFloat(utils.format.formatNearAmount(donation.total_amount));
      if (!onChainSupporters[donation.account_id]) {
        onChainSupporters[donation.account_id] = {
          account_id: donation.account_id,
          total_donated: parsed,
        };
      } else {
        onChainSupporters[donation.account_id].total_donated += parsed;
      }
    }

    // Merge with external supporters, summing donations if duplicate
    for (const ext of externalSupporters) {
      if (onChainSupporters[ext.account_id]) {
        onChainSupporters[ext.account_id].total_donated += ext.total_donated;
      } else {
        onChainSupporters[ext.account_id] = { ...ext };
      }
    }

    const mergedSupporters = Object.values(onChainSupporters).sort(
      (a, b) => b.total_donated - a.total_donated,
    );

    set({ topSupporters: mergedSupporters });
  },
}));
