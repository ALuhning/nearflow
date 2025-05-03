import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

import * as dotenv from "dotenv";
import svgr from "vite-plugin-svgr";
import tsconfigPaths from "vite-tsconfig-paths";
import polyfillNode from 'rollup-plugin-polyfill-node';
import { API_ROUTES, BASENAME, PORT, PROXY_TARGET } from "./src/customization/config-constants";
import path from 'path';
import { fileURLToPath } from 'url';

// Derive __dirname in an ES module environment
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

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
        // react: path.resolve(__dirname, 'node_modules/react'),
        // 'react-dom': path.resolve(__dirname, 'node_modules/react-dom'),
        process: 'process/browser',
        buffer: 'buffer',
        http: 'http-browserify',
        stream: 'stream-browserify',
        util: 'util',
        url: 'url',
        whatwgUrl: 'whatwg-url',
        https: 'https-browserify',
      },
    },
    define: {
      global: 'globalThis',  
      'process.env': {},  // Ensure process.env is properly defined
      'process': 'globalThis.process',  // Directly assign process to globalThis
      "process.env.BACKEND_URL": JSON.stringify(envLangflow.BACKEND_URL ?? "http://127.0.0.1:7860"),
      "process.env.ACCESS_TOKEN_EXPIRE_SECONDS": JSON.stringify(envLangflow.ACCESS_TOKEN_EXPIRE_SECONDS ?? 60),
      "process.env.CI": JSON.stringify(envLangflow.CI ?? false),
      "process.env.LANGFLOW_AUTO_LOGIN": JSON.stringify(envLangflow.LANGFLOW_AUTO_LOGIN ?? true),
    },
    plugins: [
      react(),
      svgr(), 
      tsconfigPaths(),
      polyfillNode()
    ],
    optimizeDeps: {
      esbuildOptions: {
        define: {
          global: 'globalThis',
        },
      },
    },
    server: {
      port: port,
      proxy: {
        ...proxyTargets,
      },
    }
  };
});
