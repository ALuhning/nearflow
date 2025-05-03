import { useDarkStore } from "@/stores/darkStore";
import React, { forwardRef } from "react";
import BWSvgPython from "./Python";
import { useShallow } from "zustand/react/shallow";

export const BWPythonIcon = forwardRef<
  SVGSVGElement,
  React.PropsWithChildren<{}>
>((props, ref) => {
  const isdark = useDarkStore(useShallow((state) => state.dark.toString()));
  return <BWSvgPython ref={ref} {...props} isdark={isdark} />;
});
