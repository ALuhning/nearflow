import { useDarkStore } from "@/stores/darkStore";
import React, { forwardRef } from "react";
import SvgAnthropicBox from "./Anthropic";
import { useShallow } from "zustand/react/shallow";

export const AnthropicIcon = forwardRef<
  SVGSVGElement,
  React.PropsWithChildren<{}>
>((props, ref) => {
  const isDark = useDarkStore(useShallow((state) => state.dark));

  return <SvgAnthropicBox ref={ref} {...props} isDark={isDark} />;
});
