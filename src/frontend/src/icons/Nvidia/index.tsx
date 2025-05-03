import { useDarkStore } from "@/stores/darkStore";
import React, { forwardRef } from "react";
import NvidiaSVG from "./nvidia";
import { useShallow } from "zustand/react/shallow";

export const NvidiaIcon = forwardRef<
  SVGSVGElement,
  React.PropsWithChildren<{}>
>((props, ref) => {
  const isdark = useDarkStore(useShallow((state) => state.dark)).toString();
  return <NvidiaSVG ref={ref} isdark={isdark} {...props} />;
});
