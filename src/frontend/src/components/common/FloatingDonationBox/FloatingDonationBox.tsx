import { useNavigate, useLocation } from "react-router";

export default function FloatingDonationBox() {
  const navigate = useNavigate();
  const location = useLocation();

  // Don't show on donations page
  if (location.pathname.startsWith("/donations")) {
    return null;
  }

  return (
    <div
      onClick={() => navigate("/donations")}
      className="fixed bottom-4 right-4 z-50 cursor-pointer rounded-lg bg-yellow-100 dark:bg-yellow-900 text-sm text-black dark:text-white px-4 py-2 shadow-lg hover:bg-yellow-200 dark:hover:bg-yellow-800 transition-all animate-float"
    >
      ❤️ Built by one dev. <span className="underline">Support Nearflow →</span>
    </div>
  );
}
