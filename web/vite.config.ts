import { svelte } from "@sveltejs/vite-plugin-svelte";
import { rmSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { defineConfig, loadEnv, type Plugin } from "vite";

// Cross-origin isolation lets ONNX Runtime's WebAssembly backend use threads (SharedArrayBuffer).
// Static hosts need the same two headers; see README.md.
const isolation = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
};

// With the models hosted elsewhere (VITE_MODELS_URL), a build leaves out public/models/.
function withoutLocalModels(modelsUrl: string | undefined): Plugin {
  return {
    name: "without-local-models",
    apply: "build",
    closeBundle() {
      if (modelsUrl) rmSync(fileURLToPath(new URL("dist/models", import.meta.url)), { recursive: true, force: true });
    },
  };
}

export default defineConfig(({ mode }) => ({
  plugins: [svelte(), withoutLocalModels(loadEnv(mode, process.cwd(), "VITE_").VITE_MODELS_URL)],
  base: "./",
  resolve: {
    // ORT's non-bundled build: it loads its WebAssembly glue from public/ort/ (see scripts/copy-ort.ts)
    // instead of carrying it inside the bundle.
    alias: { "onnxruntime-web": fileURLToPath(new URL("node_modules/onnxruntime-web/dist/ort.min.mjs", import.meta.url)) },
  },
  optimizeDeps: { exclude: ["onnxruntime-web"] },
  worker: { format: "es" },
  server: { headers: isolation },
  preview: { headers: isolation },
}));
