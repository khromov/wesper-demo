// End-to-end test: runs the app in Google Chrome with a fake microphone that plays
// sample_whisper.wav, and checks every path a person would use.
//   - The Swedish encoder, fp16 and WebGPU are the defaults.
//   - Push-to-talk works with the mouse and with Space, and converts with both encoders.
//   - An uploaded file converts too, and "Run again" works after switching to fp32 and to WASM.
//   - WebGPU and WASM agree at fp32, and the two encoders give different results.
//
//   bun run test:e2e            # against the dev server (needs the models: see README.md)
//   bun run test:e2e --preview  # against a production build (vite build + vite preview)
//   bun run test:e2e --headed   # watch it happen
// Each step has a timeout, and the whole run is killed after 15 minutes.
import { spawn } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { chromium, type Page } from "playwright-core";
import { decodeWav } from "../src/lib/audio/wav";

const WEB = join(import.meta.dir, "..");
const SAMPLE = join(WEB, "..", "sample_whisper.wav");
const SAMPLE_LENGTH = decodeWav(readFileSync(SAMPLE)).samples.length; // 16 kHz already
const PORT = 5199;
const STEP_MS = 5 * 60_000;

const watchdog = setTimeout(() => {
  console.error("e2e: still running after 15 minutes, giving up");
  process.exit(1);
}, 15 * 60_000);

if (!existsSync(join(WEB, "public", "models", "models.json"))) {
  console.error("e2e: no models in web/public/models. Run: .venv/bin/python web/export_models.py");
  process.exit(1);
}

const PREVIEW = process.argv.includes("--preview");
if (PREVIEW) {
  const build = Bun.spawnSync(["bunx", "vite", "build"], { cwd: WEB, timeout: 5 * 60_000 });
  if (build.exitCode !== 0) {
    console.error(`e2e: build failed\n${build.stdout}${build.stderr}`);
    process.exit(1);
  }
}
const server = spawn("bunx", ["vite", ...(PREVIEW ? ["preview"] : []), "--port", String(PORT), "--strictPort"], {
  cwd: WEB,
  stdio: ["ignore", "pipe", "pipe"],
});
let serverLog = "";
server.stdout.on("data", (d) => (serverLog += d));
server.stderr.on("data", (d) => (serverLog += d));

async function waitForServer() {
  for (let i = 0; i < 300; i++) {
    try {
      if ((await fetch(`http://localhost:${PORT}/`)).ok) return;
    } catch {
      // not up yet
    }
    await new Promise((r) => setTimeout(r, 100));
  }
  throw new Error(`dev server didn't start:\n${serverLog}`);
}

interface OutputInfo {
  encoder: string;
  precision: string;
  backend: string;
  status: string;
  error: string;
  samples: number;
  rmsDb: number;
  finite: boolean;
  ms: number;
}
interface TakeInfo {
  id: number;
  samples: number;
  levelDbfs: number;
  outputs: OutputInfo[];
}

const takes = (page: Page): Promise<TakeInfo[]> =>
  page.evaluate(() => {
    const app = (window as any).__wesper;
    return app.takes.map((t: any) => ({
      id: t.id,
      samples: t.samples.length,
      levelDbfs: t.levelDbfs,
      outputs: t.outputs.map((o: any) => {
        const s: Float32Array | null = o.samples;
        let sum = 0;
        let finite = true;
        for (const v of s ?? []) {
          sum += v * v;
          finite &&= Number.isFinite(v);
        }
        return {
          encoder: o.encoder.id, precision: o.precision, backend: o.backend, status: o.status, error: o.error,
          samples: s?.length ?? 0, rmsDb: s ? 10 * Math.log10(sum / s.length + 1e-12) : -120, finite, ms: o.encodeMs + o.decodeMs,
        };
      }),
    }));
  });

/** SNR in dB of output b against output a of the take. */
const snr = (page: Page, takeId: number, a: number, b: number): Promise<number> =>
  page.evaluate(
    ([takeId, a, b]) => {
      const t = (window as any).__wesper.takes.find((t: any) => t.id === takeId);
      const [x, y]: Float32Array[] = [t.outputs[a].samples, t.outputs[b].samples];
      let p = 0;
      let e = 0;
      for (let i = 0; i < x.length; i++) {
        p += x[i] * x[i];
        e += (x[i] - y[i]) ** 2;
      }
      return 10 * Math.log10(p / Math.max(e, 1e-30));
    },
    [takeId, a, b],
  );

async function waitForOutputs(page: Page, takeId: number, count: number) {
  await page.waitForFunction(
    ([takeId, count]) => {
      const t = (window as any).__wesper.takes.find((t: any) => t.id === takeId);
      return t && t.outputs.length >= count && t.outputs.every((o: any) => o.status === "done" || o.status === "error");
    },
    [takeId, count],
    { timeout: STEP_MS, polling: 200 },
  );
  return (await takes(page)).find((t) => t.id === takeId)!;
}

const waitForReady = (page: Page) =>
  page.waitForFunction(() => (window as any).__wesper.ready, null, { timeout: STEP_MS, polling: 200 });

function check(cond: unknown, msg: string) {
  if (!cond) throw new Error(msg);
  console.log(`  ok  ${msg}`);
}

function checkTake(t: TakeInfo, precision: string, backend: string, from = 0) {
  const outs = t.outputs.slice(from);
  check(outs.map((o) => o.encoder).join() === "sv,original", `take ${t.id}: converted with sv, then original (${outs.map((o) => o.encoder)})`);
  for (const o of outs) {
    check(o.status === "done", `take ${t.id} ${o.encoder}: done${o.error ? ` (${o.error})` : ""}`);
    check(o.precision === precision && o.backend === backend, `take ${t.id} ${o.encoder}: ran ${o.precision} on ${o.backend}`);
    // 320 per 20 ms frame, plus 8 from HiFi-GAN's upsampling (its x5 layer adds one), as in Python
    check(o.samples === Math.floor(t.samples / 320) * 320 + 8, `take ${t.id} ${o.encoder}: ${o.samples} samples for ${t.samples} in`);
    check(o.finite && o.rmsDb > -40, `take ${t.id} ${o.encoder}: audible, finite output (${o.rmsDb.toFixed(1)} dBFS RMS)`);
    console.log(`      ${o.encoder} ${o.precision} ${o.backend}: ${(o.ms / 1000).toFixed(2)} s for ${(t.samples / 16000).toFixed(1)} s of audio`);
  }
}

let failed = false;
const browser = await chromium.launch({
  channel: "chrome",
  headless: !process.argv.includes("--headed"),
  args: [
    "--use-fake-ui-for-media-stream",
    "--use-fake-device-for-media-stream",
    `--use-file-for-fake-audio-capture=${SAMPLE}`,
    "--autoplay-policy=no-user-gesture-required",
    "--enable-unsafe-webgpu",
  ],
});
try {
  await waitForServer();
  const page = await browser.newPage();
  const consoleErrors: string[] = [];
  page.on("console", (m) => m.type() === "error" && consoleErrors.push(m.text()));
  page.on("pageerror", (e) => consoleErrors.push(e.message));
  await page.goto(`http://localhost:${PORT}/`);

  console.log("defaults and model loading");
  await waitForReady(page);
  const state = await page.evaluate(() => {
    const app = (window as any).__wesper;
    return { encoder: app.settings.encoder, compare: app.settings.compare, ...app.resolved, isolated: crossOriginIsolated, threads: app.caps.threads };
  });
  check(state.encoder === "sv" && state.compare, `Swedish encoder selected, compare on`);
  check(state.backend === "webgpu" && state.precision === "fp16", `WebGPU fp16 by default (${state.backend} ${state.precision})`);
  check(state.isolated && state.threads > 1, `cross-origin isolated, ${state.threads} WASM threads`);

  console.log("push-to-talk with the mouse");
  const ptt = page.getByRole("button", { name: /Hold to whisper/ });
  await ptt.hover();
  await page.mouse.down();
  await page.waitForTimeout(2500);
  await page.mouse.up();
  await page.waitForFunction(() => {
    const app = (window as any).__wesper;
    return app.playing !== null && app.playing === app.takes[0]?.outputs[0]?.key;
  }, null, { timeout: STEP_MS, polling: 50 });
  check(true, "take 1: the Swedish result plays as soon as it's ready");
  let t = await waitForOutputs(page, 1, 2);
  check(t.samples > 1.5 * 16000 && t.samples < 4 * 16000, `take 1: ${(t.samples / 16000).toFixed(2)} s recorded`);
  check(t.levelDbfs > -60, `take 1: the fake microphone was heard (${t.levelDbfs.toFixed(1)} dBFS)`);
  checkTake(t, "fp16", "webgpu");

  console.log("push-to-talk with Space");
  await page.keyboard.down("Space");
  await page.waitForTimeout(1500);
  await page.keyboard.up("Space");
  t = await waitForOutputs(page, 2, 2);
  check(t.samples > 1.0 * 16000 && t.samples < 3 * 16000, `take 2: ${(t.samples / 16000).toFixed(2)} s recorded`);
  checkTake(t, "fp16", "webgpu");

  console.log("file upload, then fp32 and WASM");
  await page.getByTestId("file").setInputFiles(SAMPLE);
  t = await waitForOutputs(page, 3, 2);
  check(t.samples === SAMPLE_LENGTH, `take 3: the whole file (${t.samples} of ${SAMPLE_LENGTH} samples)`);
  checkTake(t, "fp16", "webgpu");

  await page.getByRole("radio", { name: /^fp32/ }).click();
  await waitForReady(page);
  await page.locator('[data-take="3"]').getByRole("button", { name: "Run again" }).click();
  t = await waitForOutputs(page, 3, 4);
  checkTake(t, "fp32", "webgpu", 2);

  await page.getByRole("radio", { name: /^WASM/ }).click();
  await waitForReady(page);
  await page.locator('[data-take="3"]').getByRole("button", { name: "Run again" }).click();
  t = await waitForOutputs(page, 3, 6);
  checkTake(t, "fp32", "wasm", 4);

  const gpuVsWasm = await snr(page, 3, 2, 4);
  check(gpuVsWasm > 50, `WebGPU and WASM agree at fp32 (Swedish: ${gpuVsWasm.toFixed(1)} dB SNR)`);
  const encoders = await snr(page, 3, 2, 3);
  check(encoders < 20, `the two encoders give different audio (${encoders.toFixed(1)} dB SNR)`);

  console.log("persistence");
  await page.reload();
  await waitForReady(page);
  const after = await page.evaluate(() => (window as any).__wesper.settings);
  check(after.precision === "fp32" && after.backend === "wasm", `settings survive a reload (${after.precision}, ${after.backend})`);

  check(consoleErrors.length === 0, `no console errors${consoleErrors.length ? `: ${consoleErrors.join(" | ")}` : ""}`);
} catch (e) {
  failed = true;
  console.error(`FAIL: ${e instanceof Error ? e.message : e}`);
} finally {
  await browser.close();
  server.kill();
  clearTimeout(watchdog);
}
console.log(failed ? "e2e failed" : "e2e passed");
process.exit(failed ? 1 : 0);
