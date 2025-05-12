import React, { forwardRef } from "react";
import { Link, LinkProps, useParams } from "react-router";
import { ENABLE_CUSTOM_PARAM } from "../feature-flags";

export const CustomLink = forwardRef<HTMLAnchorElement, LinkProps>(
  ({ to, ...props }, ref) => {
    const { customParam } = useParams();

    const newLocation =
      ENABLE_CUSTOM_PARAM && typeof to === "string" && to.startsWith("/")
        ? `/${customParam}${to}`
        : to;

    return <Link to={newLocation} {...props} ref={ref} />;
  }
);

CustomLink.displayName = "CustomLink";
