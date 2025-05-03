import { useDarkStore } from "@/stores/darkStore";
import React, { forwardRef } from "react";
import AstraSVG from "./AstraDB";
import { useShallow } from "zustand/react/shallow";

export const AstraDBIcon = forwardRef<
  SVGSVGElement,
  React.PropsWithChildren<{}>
>((props, ref) => {
  const isdark = useDarkStore(useShallow((state) => state.dark)).toString();
  return <AstraSVG ref={ref} isdark={isdark} {...props} />;
});
