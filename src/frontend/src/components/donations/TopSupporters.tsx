interface Supporter {
  account_id: string;
  total_amount: number; // in NEAR
}

interface TopSupportersProps {
  supporters: Supporter[];
}

const TIERS = [
  { name: "Champion", threshold: 50 },
  { name: "Advocate", threshold: 25 },
  { name: "Supporter", threshold: 10 },
];

export default function TopSupporters({ supporters }: TopSupportersProps) {
  const getTier = (amount: number) => {
    for (const tier of TIERS) {
      if (amount >= tier.threshold) return tier.name;
    }
    return null;
  };

  return (
    <div className="mt-6 p-4 bg-white dark:bg-background shadow rounded">
      <h4 className="text-lg font-semibold mb-4">Top Supporters</h4>
      {supporters.length === 0 ? (
        <p className="text-sm text-gray-500">Be the first to get featured!</p>
      ) : (
        <ul className="space-y-2 text-sm">
          {supporters.map((s, idx) => (
            <li key={idx} className="flex justify-between">
              <span>{s.account_id}</span>
              <span>
                {s.total_amount} Ⓝ{" "}
                <span className="ml-1 text-xs text-gray-500 italic">
                  ({getTier(s.total_amount)})
                </span>
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
