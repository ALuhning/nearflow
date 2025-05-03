import { useDarkStore } from "@/stores/darkStore";
import React, { forwardRef } from "react";
import SvgNovita from "./novita";
import { useShallow } from "zustand/react/shallow";

export const NovitaIcon = forwardRef<
  SVGSVGElement,
  React.PropsWithChildren<{}>
>((props, ref) => {
  const isdark = useDarkStore(useShallow((state) => state.dark)).toString();

  return <SvgNovita ref={ref} {...props} isdark={isdark} />;
});
