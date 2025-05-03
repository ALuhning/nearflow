import { APIClassType, ResponseErrorDetailAPI } from "@/types/api";
import { UseMutationResult } from "@tanstack/react-query";
import { useEffect } from "react";
import useAlertStore from "../../stores/alertStore";
import { mutateTemplate } from "../helpers/mutate-template";
import { useShallow } from "zustand/react/shallow";

const useFetchDataOnMount = (
  node: APIClassType,
  setNodeClass: (node: APIClassType) => void,
  name: string,
  postTemplateValue: UseMutationResult<
    APIClassType | undefined,
    ResponseErrorDetailAPI,
    any
  >,
) => {
  const setErrorData = useAlertStore(useShallow((state) => state.setErrorData));

  useEffect(() => {
    async function fetchData() {
      const template = node.template[name];
      if (
        (template?.real_time_refresh || template?.refresh_button) &&
        // options can be undefined but not an empty array
        (template?.options?.length ?? 0) === 0
      ) {
        mutateTemplate(
          template?.value,
          node,
          setNodeClass,
          postTemplateValue,
          setErrorData,
        );
      }
    }
    fetchData();
  }, []); // Empty dependency array ensures that this effect runs only once, on mount
};

export default useFetchDataOnMount;
