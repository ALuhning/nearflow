import { useDarkStore } from "@/stores/darkStore";
import React, { forwardRef } from "react";
import XAISVG from "./xAIIcon.jsx";
import { useShallow } from "zustand/react/shallow";

export const XAIIcon = forwardRef<SVGSVGElement, React.PropsWithChildren<{}>>(
  (props, ref) => {
    const isdark = useDarkStore(useShallow((state) => state.dark)).toString();
    return <XAISVG ref={ref} isdark={isdark} {...props} />;
  },
);
