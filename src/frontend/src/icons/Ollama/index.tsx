import { useDarkStore } from "@/stores/darkStore";
import React, { forwardRef } from "react";
import SvgOllama from "./Ollama";
import { useShallow } from "zustand/react/shallow";

export const OllamaIcon = forwardRef<
  SVGSVGElement,
  React.PropsWithChildren<{}>
>((props, ref) => {
  const isDark = useDarkStore(useShallow((state) => state.dark));

  return <SvgOllama ref={ref} {...props} color={isDark ? "#fff" : "#000"} />;
});
