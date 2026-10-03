import { expect, test } from "bun:test";
import { resolve, type Capabilities } from "./backend";

const gpu: Capabilities = { webgpu: true, adapter: "test", threads: 8 };

test("auto prefers WebGPU", () => {
  expect(resolve("auto", gpu)).toEqual({ backend: "webgpu", notes: [] });
  expect(resolve("wasm", gpu)).toEqual({ backend: "wasm", notes: [] });
});

test("falls back to WebAssembly without WebGPU", () => {
  expect(resolve("auto", { ...gpu, webgpu: false })).toEqual({ backend: "wasm", notes: [] });
  const forced = resolve("webgpu", { ...gpu, webgpu: false });
  expect(forced.backend).toBe("wasm");
  expect(forced.notes[0]).toContain("WebGPU isn't available");
});

test("warns when WebAssembly can't use threads", () => {
  expect(resolve("wasm", { ...gpu, threads: 1 }).notes[0]).toContain("single-threaded");
});
