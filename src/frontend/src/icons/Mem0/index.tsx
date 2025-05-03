import { useDarkStore } from "@/stores/darkStore";
import React, { forwardRef } from "react";
import SvgMem from "./SvgMem";
import { useShallow } from "zustand/react/shallow";

export const Mem0 = forwardRef<SVGSVGElement, React.PropsWithChildren<{}>>(
  (props, ref) => {
    const isdark = useDarkStore(useShallow((state) => state.dark)).toString();
    return <SvgMem className="icon" ref={ref} {...props} isdark={isdark} />;
  },
);
