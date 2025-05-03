import { useShortcutsStore } from "@/stores/shortcuts";
import { useHotkeys } from "react-hotkeys-hook";
import isWrappedWithClass from "../../PageComponent/utils/is-wrapped-with-class";
import { useShallow } from "zustand/react/shallow";

export default function useShortcuts({
  showOverrideModal,
  showModalAdvanced,
  openModal,
  showconfirmShare,
  FreezeAllVertices,
  downloadFunction,
  displayDocs,
  saveComponent,
  showAdvance,
  handleCodeModal,
  shareComponent,
  ungroup,
  minimizeFunction,
  activateToolMode,
  hasToolMode,
}: {
  showOverrideModal?: boolean;
  showModalAdvanced?: boolean;
  openModal?: boolean;
  showconfirmShare?: boolean;
  FreezeAllVertices?: () => void;
  downloadFunction?: () => void;
  displayDocs?: () => void;
  saveComponent?: () => void;
  showAdvance?: () => void;
  handleCodeModal?: () => void;
  shareComponent?: () => void;
  ungroup?: () => void;
  minimizeFunction?: () => void;
  activateToolMode?: () => void;
  hasToolMode?: boolean;
}) {
  const advancedSettings = useShortcutsStore(useShallow((state) => state.advancedSettings));
  const minimize = useShortcutsStore(useShallow((state) => state.minimize));
  const componentShare = useShortcutsStore(useShallow((state) => state.componentShare));
  const save = useShortcutsStore(useShallow((state) => state.saveComponent));
  const docs = useShortcutsStore(useShallow((state) => state.docs));
  const code = useShortcutsStore(useShallow((state) => state.code));
  const group = useShortcutsStore(useShallow((state) => state.group));
  const download = useShortcutsStore(useShallow((state) => state.download));
  const freezeAll = useShortcutsStore(useShallow((state) => state.freezePath));
  const toolMode = useShortcutsStore(useShallow((state) => state.toolMode));

  function handleFreezeAll(e: KeyboardEvent) {
    if (isWrappedWithClass(e, "noflow") || !FreezeAllVertices) return;
    e.preventDefault();
    FreezeAllVertices();
  }

  function handleDownloadWShortcut(e: KeyboardEvent) {
    if (!downloadFunction) return;
    e.preventDefault();
    downloadFunction();
  }

  function handleDocsWShortcut(e: KeyboardEvent) {
    if (!displayDocs) return;
    e.preventDefault();
    displayDocs();
  }

  function handleSaveWShortcut(e: KeyboardEvent) {
    if (
      (isWrappedWithClass(e, "noflow") && !showOverrideModal) ||
      !saveComponent
    )
      return;
    e.preventDefault();
    saveComponent();
  }

  function handleAdvancedWShortcut(e: KeyboardEvent) {
    if ((isWrappedWithClass(e, "noflow") && !showModalAdvanced) || !showAdvance)
      return;
    e.preventDefault();
    showAdvance();
  }

  function handleCodeWShortcut(e: KeyboardEvent) {
    if ((isWrappedWithClass(e, "noflow") && !openModal) || !handleCodeModal)
      return;
    e.preventDefault();
    handleCodeModal();
  }

  function handleShareWShortcut(e: KeyboardEvent) {
    if (
      (isWrappedWithClass(e, "noflow") && !showconfirmShare) ||
      !shareComponent
    )
      return;
    e.preventDefault();
    shareComponent();
  }

  function handleGroupWShortcut(e: KeyboardEvent) {
    if (isWrappedWithClass(e, "noflow") || !ungroup) return;
    e.preventDefault();
    ungroup();
  }

  function handleMinimizeWShortcut(e: KeyboardEvent) {
    if (isWrappedWithClass(e, "noflow") || !minimizeFunction) return;
    e.preventDefault();
    minimizeFunction();
  }

  function handleToolModeWShortcut(e: KeyboardEvent, hasToolMode?: boolean) {
    if (!hasToolMode) return;
    if (isWrappedWithClass(e, "noflow") || !activateToolMode) return;
    e.preventDefault();
    activateToolMode();
  }

  useHotkeys(minimize, handleMinimizeWShortcut, { preventDefault: true });
  useHotkeys(group, handleGroupWShortcut, { preventDefault: true });
  useHotkeys(componentShare, handleShareWShortcut, { preventDefault: true });
  useHotkeys(code, handleCodeWShortcut, { preventDefault: true });
  useHotkeys(advancedSettings, handleAdvancedWShortcut, {
    preventDefault: true,
  });
  useHotkeys(save, handleSaveWShortcut, { preventDefault: true });
  useHotkeys(docs, handleDocsWShortcut, { preventDefault: true });
  useHotkeys(download, handleDownloadWShortcut, { preventDefault: true });
  useHotkeys(freezeAll, handleFreezeAll);
  useHotkeys(toolMode, (e) => handleToolModeWShortcut(e, hasToolMode), {
    preventDefault: true,
  });
}
