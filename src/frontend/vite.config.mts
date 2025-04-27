import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "path";
import * as dotenv from "dotenv";
import svgr from "vite-plugin-svgr";
import tsconfigPaths from "vite-tsconfig-paths";
import { API_ROUTES, BASENAME, PORT, PROXY_TARGET } from "./src/customization/config-constants";

// Polyfill for buffer
import { Buffer } from 'buffer';

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");

  const envLangflowResult = dotenv.config({
    path: path.resolve(__dirname, "../../.env"),
  });

  const envLangflow = envLangflowResult.parsed || {};

  const apiRoutes = API_ROUTES || ["^/api/v1/", "^/api/v2/", "/health"];

  const target = env.VITE_PROXY_TARGET || PROXY_TARGET || "http://127.0.0.1:7860";

  const port = Number(env.VITE_PORT) || PORT || 3000;

  const proxyTargets = apiRoutes.reduce((proxyObj, route) => {
    proxyObj[route] = {
      target: target,
      changeOrigin: true,
      secure: false,
      ws: true,
    };
    return proxyObj;
  }, {});

  return {
    base: BASENAME || "",
    build: {
      outDir: "build",
    },
    resolve: {
      alias: {
        buffer: path.resolve(__dirname, 'node_modules', 'buffer'), // Correct way to resolve path for buffer
      },
    },
    define: {
      // Provide the buffer polyfill
      "global.Buffer": JSON.stringify(Buffer),
      "process.env.BACKEND_URL": JSON.stringify(envLangflow.BACKEND_URL ?? "http://127.0.0.1:7860"),
      "process.env.ACCESS_TOKEN_EXPIRE_SECONDS": JSON.stringify(envLangflow.ACCESS_TOKEN_EXPIRE_SECONDS ?? 60),
      "process.env.CI": JSON.stringify(envLangflow.CI ?? false),
      "process.env.LANGFLOW_AUTO_LOGIN": JSON.stringify(envLangflow.LANGFLOW_AUTO_LOGIN ?? true),
    },
    plugins: [react(), svgr(), tsconfigPaths()],
    server: {
      port: port,
      proxy: {
        ...proxyTargets,
      },
    },
  };
});
