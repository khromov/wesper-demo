import { expect, test } from "bun:test";
import { resolve, type Capabilities } from "./backend";

const gpu: Capabilities = { webgpu: true, shaderF16: true, adapter: "test", threads: 8 };

test("auto prefers WebGPU and keeps the requested precision", () => {
  expect(resolve("auto", "fp16", gpu)).toEqual({ backend: "webgpu", precision: "fp16", notes: [] });
  expect(resolve("auto", "fp32", gpu)).toEqual({ backend: "webgpu", precision: "fp32", notes: [] });
});

test("falls back to WebAssembly without WebGPU", () => {
  const r = resolve("auto", "fp16", { ...gpu, webgpu: false });
  expect([r.backend, r.precision, r.notes]).toEqual(["wasm", "fp16", []]);
  const forced = resolve("webgpu", "fp16", { ...gpu, webgpu: false });
  expect(forced.backend).toBe("wasm");
  expect(forced.notes[0]).toContain("WebGPU isn't available");
});

test("uses fp32 on a GPU without fp16 shaders", () => {
  const r = resolve("auto", "fp16", { ...gpu, shaderF16: false });
  expect([r.backend, r.precision]).toEqual(["webgpu", "fp32"]);
  expect(r.notes[0]).toContain("fp16 shaders");
  expect(resolve("wasm", "fp16", { ...gpu, shaderF16: false }).precision).toBe("fp16");
});

test("warns when WebAssembly can't use threads", () => {
  expect(resolve("wasm", "fp32", { ...gpu, threads: 1 }).notes[0]).toContain("single-threaded");
});
