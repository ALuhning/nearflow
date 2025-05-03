import { useDarkStore } from "@/stores/darkStore";
import React, { forwardRef } from "react";
import HCDSVG from "./HCD";
import { useShallow } from "zustand/react/shallow";

export const HCDIcon = forwardRef<SVGSVGElement, React.PropsWithChildren<{}>>(
  (props, ref) => {
    const isdark = useDarkStore(useShallow((state) => state.dark)).toString();

    return <HCDSVG ref={ref} isdark={isdark} {...props} />;
  },
);
