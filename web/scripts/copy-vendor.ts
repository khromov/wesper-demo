// Copies files that are served as-is rather than bundled into public/. Runs on `bun install`.
// - ONNX Runtime Web's WebAssembly files, to public/ort/, where the worker loads them from
//   (ort.env.wasm.wasmPaths). The .mjs starts its own worker threads from its own URL, which
//   bundling would break. Only the JSEP build (WebGPU and WASM): the newer native WebGPU build
//   (*.asyncify.*) gave NaN audio from the decoder in Chrome 154.
// - coi-serviceworker, to public/. On hosts that can't send the COOP/COEP headers (GitHub Pages),
//   it adds them, so WASM can use threads. It must be a file of its own: it registers itself.
import { copyFileSync, mkdirSync, readdirSync, rmSync } from "node:fs";
import { join } from "node:path";

const root = join(import.meta.dir, "..");
const ortSrc = join(root, "node_modules", "onnxruntime-web", "dist");
const ortDst = join(root, "public", "ort");
const ort = readdirSync(ortSrc).filter((f) => /^ort-wasm-simd-threaded\.jsep\.(mjs|wasm)$/.test(f));
if (ort.length !== 2) throw new Error(`expected the JSEP .mjs and .wasm in ${ortSrc}, found ${ort.join(", ") || "none"}`);
rmSync(ortDst, { recursive: true, force: true });
mkdirSync(ortDst, { recursive: true });
for (const f of ort) copyFileSync(join(ortSrc, f), join(ortDst, f));

copyFileSync(join(root, "node_modules", "coi-serviceworker", "coi-serviceworker.min.js"), join(root, "public", "coi-serviceworker.min.js"));
console.log(`copied ${ort.join(", ")} to public/ort/, and coi-serviceworker.min.js to public/`);
