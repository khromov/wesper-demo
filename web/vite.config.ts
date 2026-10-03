import { svelte } from "@sveltejs/vite-plugin-svelte";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

// Cross-origin isolation lets ONNX Runtime's WebAssembly backend use threads (SharedArrayBuffer).
// Static hosts need the same two headers; see README.md.
const isolation = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
};

export default defineConfig({
  plugins: [svelte()],
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
});
