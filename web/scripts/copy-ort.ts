// Copies ONNX Runtime Web's WebAssembly files to public/ort/, where the worker loads them from
// (ort.env.wasm.wasmPaths). They're served as-is rather than bundled: the .mjs starts its own
// worker threads from its own URL, which bundling would break. Runs on `bun install`.
import { copyFileSync, mkdirSync, readdirSync, rmSync } from "node:fs";
import { join } from "node:path";

const src = join(import.meta.dir, "..", "node_modules", "onnxruntime-web", "dist");
const dst = join(import.meta.dir, "..", "public", "ort");
// The JSEP build: WebGPU and WASM. (The newer native WebGPU build, *.asyncify.*, gave NaN audio
// from the fp32 decoder in Chrome 154.)
const files = readdirSync(src).filter((f) => /^ort-wasm-simd-threaded\.jsep\.(mjs|wasm)$/.test(f));
if (files.length !== 2) throw new Error(`expected the JSEP .mjs and .wasm in ${src}, found ${files.join(", ") || "none"}`);
rmSync(dst, { recursive: true, force: true });
mkdirSync(dst, { recursive: true });
for (const f of files) copyFileSync(join(src, f), join(dst, f));
console.log(`copied ${files.join(", ")} to public/ort/`);
