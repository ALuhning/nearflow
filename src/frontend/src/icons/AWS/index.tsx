import { useDarkStore } from "@/stores/darkStore";
import React, { forwardRef } from "react";
import SvgAWS from "./AWS";
import { useShallow } from "zustand/react/shallow";

export const AWSIcon = forwardRef<SVGSVGElement, React.PropsWithChildren<{}>>(
  (props, ref) => {
    const isdark = useDarkStore(useShallow((state) => state.dark)).toString();
    return <SvgAWS ref={ref} isdark={isdark} {...props} />;
  },
);
