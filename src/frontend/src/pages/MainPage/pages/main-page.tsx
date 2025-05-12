import { JSX } from "react";
import SideBarFoldersButtonsComponent from "@/components/core/folderSidebarComponent/components/sideBarFolderButtons";
import { SidebarProvider } from "@/components/ui/sidebar";
import { useDeleteFolders } from "@/controllers/API/queries/folders";
import CustomLoader from "@/customization/components/custom-loader";
import { useCustomNavigate } from "@/customization/hooks/use-custom-navigate";
import useAlertStore from "@/stores/alertStore";
import useFlowsManagerStore from "@/stores/flowsManagerStore";
import { useFolderStore } from "@/stores/foldersStore";
import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { Outlet } from "react-router";
import ModalsComponent from "../components/modalsComponent";
import EmptyPage from "./emptyPage";
import { useShallow } from "zustand/react/shallow";
import EmptyPageCommunity from "./empty-page";

export default function CollectionPage(): JSX.Element {
  const [openModal, setOpenModal] = useState(false);
  const [openDeleteFolderModal, setOpenDeleteFolderModal] = useState(false);
  const setFolderToEdit = useFolderStore(useShallow((state) => state.setFolderToEdit));
  const navigate = useCustomNavigate();
  const flows = useFlowsManagerStore(useShallow((state) => state.flows));
  const examples = useFlowsManagerStore(useShallow((state) => state.examples));
  const setSuccessData = useAlertStore(useShallow((state) => state.setSuccessData));
  const setErrorData = useAlertStore(useShallow((state) => state.setErrorData));
  const folderToEdit = useFolderStore(useShallow((state) => state.folderToEdit));
  const folders = useFolderStore(useShallow((state) => state.folders));
  const queryClient = useQueryClient();

  useEffect(() => {
    return () => queryClient.removeQueries({ queryKey: ["useGetFolder"] });
  }, []);

  const { mutate } = useDeleteFolders();

  const handleDeleteFolder = () => {
    mutate(
      {
        folder_id: folderToEdit?.id!,
      },
      {
        onSuccess: () => {
          setSuccessData({
            title: "Project deleted successfully.",
          });
          navigate("/all");
        },
        onError: (err) => {
          console.error(err);
          setErrorData({
            title: "Error deleting project.",
          });
        },
      },
    );
  };

  return (
    <SidebarProvider width="280px">
      {flows &&
        examples &&
        folders &&
        (flows?.length !== examples?.length || folders?.length > 1) && (
          <SideBarFoldersButtonsComponent
            handleChangeFolder={(id: string) => {
              navigate(`all/folder/${id}`);
            }}
            handleDeleteFolder={(item) => {
              setFolderToEdit(item);
              setOpenDeleteFolderModal(true);
            }}
            handleFilesClick={() => {
              navigate("files");
            }}
          />
        )}
      <main className="flex h-full w-full overflow-hidden">
        {flows && examples && folders ? (
          <div
            className={`relative mx-auto flex h-full w-full flex-col overflow-hidden`}
          >
            {flows?.length !== examples?.length || folders?.length > 1 ? (
              <Outlet />
            ) : (
              // <EmptyPage setOpenModal={setOpenModal} />

              <EmptyPageCommunity setOpenModal={setOpenModal} />
            )}
          </div>
        ) : (
          <div className="flex h-full w-full items-center justify-center">
            <CustomLoader remSize={30} />
          </div>
        )}
      </main>
      <ModalsComponent
        openModal={openModal}
        setOpenModal={setOpenModal}
        openDeleteFolderModal={openDeleteFolderModal}
        setOpenDeleteFolderModal={setOpenDeleteFolderModal}
        handleDeleteFolder={handleDeleteFolder}
      />
    </SidebarProvider>
  );
}
