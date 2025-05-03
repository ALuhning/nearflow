import "@xyflow/react/dist/style.css";
import { Suspense, lazy, useEffect } from "react";
import { RouterProvider } from "react-router";
import { LoadingPage } from "./pages/LoadingPage";
import router from "./routes";
import { useDarkStore } from "./stores/darkStore";
import { useShallow } from "zustand/react/shallow";
import '@near-wallet-selector/modal-ui/styles.css';

export default function App() {
  const dark = useDarkStore(useShallow((state) => state.dark));

  // Initialize wallet selector on component mount
  useEffect(() => {
    if (!dark) {
      document.getElementById("body")!.classList.remove("dark");
    } else {
      document.getElementById("body")!.classList.add("dark");
    }
  }, [dark]);

  return (
    <>
        <Suspense fallback={<LoadingPage />}>
          <RouterProvider router={router}/>
        </Suspense>
    </>
  );
}
