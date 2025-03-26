import React, { forwardRef } from "react";
import SvgNearIcon from "./nearIcon";

export const NearIcon = forwardRef<
  SVGSVGElement,
  React.PropsWithChildren<{}>
>((props, ref) => {
  return <SvgNearIcon ref={ref} {...props} />;
});