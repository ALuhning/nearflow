import useFlowsManagerStore from "@/stores/flowsManagerStore";
import { FlowType } from "@/types/flow";
import { useDebounce } from "../use-debounce";
import useSaveFlow from "./use-save-flow";
import { useShallow } from "zustand/react/shallow";

const useAutoSaveFlow = () => {
  const saveFlow = useSaveFlow();
  const autoSaving = useFlowsManagerStore(useShallow((state) => state.autoSaving));
  const autoSavingInterval = useFlowsManagerStore(useShallow(
    (state) => state.autoSavingInterval,
  ));

  const autoSaveFlow = useDebounce((flow?: FlowType) => {
    if (autoSaving) {
      saveFlow(flow);
    }
  }, autoSavingInterval);

  return autoSaveFlow;
};

export default useAutoSaveFlow;
