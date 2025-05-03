import useFlowStore from "../stores/flowStore";
import useFlowsManagerStore from "../stores/flowsManagerStore";
import { customStringify } from "../utils/reactflowUtils";
import { useShallow } from "zustand/react/shallow";

export function useUnsavedChanges() {
  const currentFlow = useFlowStore(useShallow((state) => state.currentFlow));
  const savedFlow = useFlowsManagerStore(useShallow((state) => state.currentFlow));

  if (!currentFlow || !savedFlow) {
    return false;
  }

  return customStringify(currentFlow) !== customStringify(savedFlow);
}
