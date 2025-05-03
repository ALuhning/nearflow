import { useDarkStore } from "@/stores/darkStore";
import React, { forwardRef } from "react";
import SvgJSIcon from "./JSIcon";
import { useShallow } from "zustand/react/shallow";

export const JSIcon = forwardRef<SVGSVGElement, React.PropsWithChildren<{}>>(
  (props, ref) => {
    const isdark = useDarkStore(useShallow((state) => state.dark.toString()));
    return <SvgJSIcon ref={ref} {...props} isdark={isdark} />;
  },
);
