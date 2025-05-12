import { useGetAutoLogin } from "@/controllers/API/queries/auth";
import { useGetConfig } from "@/controllers/API/queries/config/use-get-config";
import { useGetBasicExamplesQuery } from "@/controllers/API/queries/flows/use-get-basic-examples";
import { useGetFoldersQuery } from "@/controllers/API/queries/folders/use-get-folders";
import { useGetTagsQuery } from "@/controllers/API/queries/store";
import { useGetGlobalVariables } from "@/controllers/API/queries/variables";
import { useGetVersionQuery } from "@/controllers/API/queries/version";
import { CustomLoadingPage } from "@/customization/components/custom-loading-page";
import { useCustomPrimaryLoading } from "@/customization/hooks/use-custom-primary-loading";
import { useDarkStore } from "@/stores/darkStore";
import useFlowsManagerStore from "@/stores/flowsManagerStore";
import { useEffect } from "react";
import { Outlet } from "react-router";
import { LoadingPage } from "../LoadingPage";
import { useShallow } from "zustand/react/shallow";
import { IS_AUTO_LOGIN } from "@/constants/constants";

export function AppInitPage() {
  const refreshStars = useDarkStore(useShallow((state) => state.refreshStars));
  const refreshDiscordCount = useDarkStore(useShallow(
    (state) => state.refreshDiscordCount,
  ));
  const isLoading = useFlowsManagerStore(useShallow((state) => state.isLoading));

  const { isFetched: isLoaded } = useCustomPrimaryLoading();

  const autoLoginQuery = IS_AUTO_LOGIN
  ? useGetAutoLogin({ enabled: false })
  : { isFetched: true, refetch: () => {} };
  const { isFetched, refetch } = autoLoginQuery;
  useGetVersionQuery({ enabled: isFetched });
  const { isFetched: isConfigFetched } = useGetConfig({ enabled: isFetched });
  useGetGlobalVariables({ enabled: isFetched });
  useGetTagsQuery({ enabled: isFetched });
  useGetFoldersQuery({ enabled: isFetched });
  const { isFetched: isExamplesFetched, refetch: refetchExamples } =
    useGetBasicExamplesQuery();

  useEffect(() => {
    if (IS_AUTO_LOGIN && isLoaded && !isFetched) {
      refetch();
    }
  }, [IS_AUTO_LOGIN, isLoaded, isFetched, refetch]);

  useEffect(() => {
    if (isFetched) {
      refreshStars();
      refreshDiscordCount();
    }

    if (isConfigFetched) {
      refetch();
      refetchExamples();
    }
  }, [isFetched, isConfigFetched]);

  return (
    //need parent component with width and height
    <>
      {isLoaded ? (
        (isLoading || !isFetched || !isExamplesFetched) && (
          <LoadingPage overlay />
        )
      ) : (
        <CustomLoadingPage />
      )}
      {isFetched && isExamplesFetched && <Outlet />}
    </>
  );
}
