// End-to-end test: runs the app in Google Chrome with a fake microphone that plays
// sample_whisper.wav, and checks every path a person would use.
//   - The Swedish encoder, the Swedish narrator voice and WebGPU are the defaults.
//   - Push-to-talk works with the mouse and with Space, and converts with both encoders.
//   - Comparing voices converts with every encoder and voice in models.json, each voice at its
//     own sample rate (BigVGAN's 22.05 kHz too), on WASM and WebGPU, and the voices differ.
//   - An uploaded file converts too, and "Run again" works after switching to WASM.
//   - WebGPU and WASM agree, and the two encoders give different results.
//   - Old downloads are cleaned up, and a private window says it can't keep the models.
//
//   bun run test:e2e            # against the dev server (needs the models: see README.md)
//   bun run test:e2e --preview  # against a production build (vite build + vite preview)
//   bun run test:e2e --pages    # a production build set up like GitHub Pages + models on S3
//   bun run test:e2e --models DIR  # with the models in DIR (an export_models.py --out), served
//                                  # from another origin, as with --pages
//   bun run test:e2e --headed   # watch it happen
// Each step has a timeout, and the whole run is killed after 20 minutes.
import { spawn } from "node:child_process";
import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { chromium, type Page } from "playwright-core";
import { decodeWav } from "../src/lib/audio/wav";

const WEB = join(import.meta.dir, "..");
const SAMPLE = join(WEB, "..", "sample_whisper.wav");
const SAMPLE_LENGTH = decodeWav(readFileSync(SAMPLE)).samples.length; // 16 kHz already
const PORT = 5199;
const STEP_MS = 5 * 60_000;

const watchdog = setTimeout(() => {
  console.error("e2e: still running after 20 minutes, giving up");
  process.exit(1);
}, 20 * 60_000);

const modelsArg = process.argv.indexOf("--models");
const MODELS_DIR = modelsArg > 0 ? resolve(process.argv[modelsArg + 1] ?? "") : join(WEB, "public", "models");
if (!existsSync(join(MODELS_DIR, "models.json"))) {
  console.error(`e2e: no models in ${MODELS_DIR}. Run: .venv/bin/python web/export_models.py`);
  process.exit(1);
}
interface VoiceInfo {
  id: string;
  sampleRate?: number;
  hop?: number;
}
const VOICES: VoiceInfo[] = JSON.parse(readFileSync(join(MODELS_DIR, "models.json"), "utf8")).decoders;

// dev: Vite's dev server. preview: a production build on Vite's preview server. pages: a
// production build as GitHub Pages serves it, under /wesper-demo/ and without the COOP/COEP
// headers, with the models on another origin that sends CORS headers, as S3 does.
const MODE = process.argv.includes("--pages") ? "pages" : process.argv.includes("--preview") ? "preview" : "dev";
const MODELS_PORT = 5299;
const APP_URL = `http://localhost:${PORT}/${MODE === "pages" ? "wesper-demo/" : ""}`;
// The models on their own origin, as on S3: for --pages, and for --models in any mode.
const MODELS_ELSEWHERE = MODE === "pages" || modelsArg > 0;
const modelsEnv: Record<string, string> = MODELS_ELSEWHERE ? { VITE_MODELS_URL: `http://localhost:${MODELS_PORT}/` } : {};

function build(env: Record<string, string> = {}) {
  const r = Bun.spawnSync(["bunx", "vite", "build"], { cwd: WEB, env: { ...process.env, ...env }, timeout: 5 * 60_000 });
  if (r.exitCode !== 0) {
    console.error(`e2e: build failed\n${r.stdout}${r.stderr}`);
    process.exit(1);
  }
}

/** Serves the files in dir under the URL path prefix, with extra headers. */
function serveStatic(port: number, dir: string, prefix: string, headers: Record<string, string> = {}) {
  return Bun.serve({
    port,
    async fetch(req) {
      const path = decodeURIComponent(new URL(req.url).pathname);
      const file = Bun.file(join(dir, path.slice(prefix.length) || "index.html"));
      if (!path.startsWith(prefix) || !(await file.exists())) return new Response("not found", { status: 404, headers });
      return new Response(file, { headers });
    },
  });
}

let serverLog = "";
const stops: (() => void)[] = [];
const stopServers = () => stops.forEach((stop) => stop());
if (MODELS_ELSEWHERE) {
  const models = serveStatic(MODELS_PORT, MODELS_DIR, "/", { "Access-Control-Allow-Origin": "*" });
  stops.push(() => models.stop(true));
}
if (MODE === "pages") {
  build(modelsEnv);
  if (existsSync(join(WEB, "dist", "models"))) {
    console.error("e2e: the build still contains models/, though they're hosted elsewhere");
    process.exit(1);
  }
  const site = serveStatic(PORT, join(WEB, "dist"), "/wesper-demo/");
  stops.push(() => site.stop(true));
} else {
  if (MODE === "preview") build(modelsEnv);
  const server = spawn("bunx", ["vite", ...(MODE === "preview" ? ["preview"] : []), "--port", String(PORT), "--strictPort"], {
    cwd: WEB,
    env: { ...process.env, ...modelsEnv },
    stdio: ["ignore", "pipe", "pipe"],
  });
  server.stdout.on("data", (d) => (serverLog += d));
  server.stderr.on("data", (d) => (serverLog += d));
  stops.push(() => server.kill());
}

/** Opens the app. On the Pages setup, coi-serviceworker reloads it once to add the headers;
 *  with `isolated`, this waits for that. */
async function open(page: Page, isolated = true) {
  await page.goto(APP_URL);
  if (MODE !== "pages" || !isolated) return;
  for (let i = 0; i < 150; i++) {
    try {
      if (await page.evaluate(() => crossOriginIsolated)) return;
    } catch {
      // the page is reloading
    }
    await new Promise((r) => setTimeout(r, 200));
  }
  throw new Error("coi-serviceworker didn't make the page cross-origin isolated");
}

async function waitForServer() {
  for (let i = 0; i < 300; i++) {
    try {
      if ((await fetch(APP_URL)).ok) return;
    } catch {
      // not up yet
    }
    await new Promise((r) => setTimeout(r, 100));
  }
  throw new Error(`dev server didn't start:\n${serverLog}`);
}

interface OutputInfo {
  encoder: string;
  decoder: string;
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
          encoder: o.encoder.id, decoder: o.decoder.id, backend: o.backend, status: o.status, error: o.error,
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

/** Waits until the models are loaded, through any reload in between. */
async function waitForReady(page: Page) {
  const until = Date.now() + STEP_MS;
  while (Date.now() < until) {
    try {
      if (await page.evaluate(() => (window as any).__wesper?.ready)) return;
    } catch {
      // the page is reloading
    }
    await new Promise((r) => setTimeout(r, 200));
  }
  throw new Error("the models didn't load");
}

function check(cond: unknown, msg: string) {
  if (!cond) throw new Error(msg);
  console.log(`  ok  ${msg}`);
}

/** How many samples a voice makes from n samples of 16 kHz input, as in Python. */
function outputLength(voice: string, n: number) {
  const v = VOICES.find((d) => d.id === voice)!;
  const [rate, hop] = [v.sampleRate ?? 16000, v.hop ?? 320];
  const units = Math.floor(n / 320);
  // HiFi-GAN 16 kHz: 320 per 20 ms unit, plus 8 from its upsampling (its x5 layer adds one)
  if (rate === 16000 && hop === 320) return units * 320 + 8;
  // others: the units moved to the vocoder's frames (vocoders.n_frames), hop samples each
  return Math.floor((units * 320 * rate) / (16000 * hop)) * hop;
}

function checkTake(t: TakeInfo, backend: string, from = 0, pairs = ["sv:sv-narrator", "original:sv-narrator"]) {
  const outs = t.outputs.slice(from);
  const got = outs.map((o) => `${o.encoder}:${o.decoder}`);
  check(got.join() === pairs.join(), `take ${t.id}: converted as ${pairs.join(", then ")} (${got})`);
  for (const o of outs) {
    const name = `take ${t.id} ${o.encoder}:${o.decoder}`;
    check(o.status === "done", `${name}: done${o.error ? ` (${o.error})` : ""}`);
    check(o.backend === backend, `${name}: ran on ${o.backend}`);
    check(o.samples === outputLength(o.decoder, t.samples), `${name}: ${o.samples} samples for ${t.samples} in`);
    check(o.finite && o.rmsDb > -40, `${name}: audible, finite output (${o.rmsDb.toFixed(1)} dBFS RMS)`);
    console.log(`      ${o.encoder}:${o.decoder} ${o.backend}: ${(o.ms / 1000).toFixed(2)} s for ${(t.samples / 16000).toFixed(1)} s of audio`);
  }
}

let failed = false;
// A real (temporary) profile, so storage behaves as for a person: the default context is
// incognito-like, with an in-memory quota too small for the models.
const profile = mkdtempSync(join(tmpdir(), "wesper-e2e-"));
const browser = await chromium.launchPersistentContext(profile, {
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
  const page = browser.pages()[0] ?? (await browser.newPage());
  const consoleErrors: string[] = [];
  page.on("console", (m) => (m.type() === "error" || m.text().includes("not caching")) && consoleErrors.push(m.text()));
  page.on("pageerror", (e) => consoleErrors.push(e.message));
  await open(page);

  console.log("defaults and model loading");
  await waitForReady(page);
  const state = await page.evaluate(() => {
    const app = (window as any).__wesper;
    return { encoder: app.settings.encoder, decoder: app.settings.decoder, compare: app.settings.compare,
             compareVoices: app.settings.compareVoices, ...app.resolved, isolated: crossOriginIsolated, threads: app.caps.threads };
  });
  check(state.encoder === "sv" && state.compare, `Swedish encoder selected, compare on`);
  check(state.decoder === "sv-narrator" && !state.compareVoices, `Swedish narrator voice selected, voice compare off`);
  check(state.backend === "webgpu", `WebGPU by default (${state.backend})`);
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
  checkTake(t, "webgpu");

  console.log("push-to-talk with Space");
  await page.keyboard.down("Space");
  await page.waitForTimeout(1500);
  await page.keyboard.up("Space");
  t = await waitForOutputs(page, 2, 2);
  check(t.samples > 1.0 * 16000 && t.samples < 3 * 16000, `take 2: ${(t.samples / 16000).toFixed(2)} s recorded`);
  checkTake(t, "webgpu");

  console.log("file upload, then WASM");
  await page.getByTestId("file").setInputFiles(SAMPLE);
  t = await waitForOutputs(page, 3, 2);
  check(t.samples === SAMPLE_LENGTH, `take 3: the whole file (${t.samples} of ${SAMPLE_LENGTH} samples)`);
  checkTake(t, "webgpu");

  await page.getByRole("radio", { name: /^WASM/ }).click();
  await waitForReady(page);
  await page.locator('[data-take="3"]').getByRole("button", { name: "Run again" }).click();
  t = await waitForOutputs(page, 3, 4);
  checkTake(t, "wasm", 2);

  const gpuVsWasm = await snr(page, 3, 0, 2);
  check(gpuVsWasm > 50, `WebGPU and WASM agree (Swedish: ${gpuVsWasm.toFixed(1)} dB SNR)`);
  const encoders = await snr(page, 3, 0, 1);
  check(encoders < 20, `the two encoders give different audio (${encoders.toFixed(1)} dB SNR)`);

  console.log("persistence and cache cleanup");
  const cached = () => page.evaluate(async () => (await (await caches.open("wesper-models-v1")).keys()).map((r) => new URL(r.url).pathname));
  const before = await cached();
  check(before.length === 3, `the three models are cached (${before.join(", ")})`);
  await page.evaluate(() => caches.open("wesper-models-v1").then((c) => c.put("/models/encoder-sv.fp16.onnx?v=old", new Response("old"))));
  await page.reload();
  await waitForReady(page);
  const after = await page.evaluate(() => (window as any).__wesper.settings);
  check(after.backend === "wasm", `settings survive a reload (${after.backend})`);
  await page.waitForFunction(
    async () => (await (await caches.open("wesper-models-v1")).keys()).every((r) => !r.url.includes("fp16")),
    null, { timeout: 30_000, polling: 200 },
  );
  check((await cached()).sort().join() === before.sort().join(), "a download no longer in models.json is deleted, the models stay");

  console.log(`comparing voices: ${VOICES.map((v) => v.id).join(", ")}`);
  await page.getByLabel(/Also convert with the .* voice/).check();
  await waitForReady(page);
  await page.getByTestId("file").setInputFiles(SAMPLE); // the reload above cleared the takes: a new one
  await page.waitForFunction(() => (window as any).__wesper.takes.length > 0, null, { timeout: STEP_MS });
  const voiceTake: number = await page.evaluate(() => (window as any).__wesper.takes[0].id);
  const voiceOrder = ["sv-narrator", ...VOICES.map((v) => v.id).filter((id) => id !== "sv-narrator")];
  const voicePairs = ["sv", "original"].flatMap((e) => voiceOrder.map((d) => `${e}:${d}`));
  t = await waitForOutputs(page, voiceTake, voicePairs.length);
  checkTake(t, "wasm", 0, voicePairs);
  for (let i = 1; i < voiceOrder.length; i++) {
    if (t.outputs[i].samples !== t.outputs[0].samples) continue; // another sample rate: different anyway
    const voices = await snr(page, voiceTake, 0, i);
    check(voices < 20, `the ${voiceOrder[0]} and ${voiceOrder[i]} voices give different audio (${voices.toFixed(1)} dB SNR)`);
  }

  console.log("comparing voices on WebGPU");
  await page.getByRole("radio", { name: /^WebGPU/ }).click();
  await waitForReady(page);
  await page.locator(`[data-take="${voiceTake}"]`).getByRole("button", { name: "Run again" }).click();
  t = await waitForOutputs(page, voiceTake, 2 * voicePairs.length);
  checkTake(t, "webgpu", voicePairs.length, voicePairs);
  for (let i = 0; i < voiceOrder.length; i++) {
    const agree = await snr(page, voiceTake, i, voicePairs.length + i);
    check(agree > 50, `WebGPU and WASM agree (${voicePairs[i]}: ${agree.toFixed(1)} dB SNR)`);
  }

  check(consoleErrors.length === 0, `no console errors${consoleErrors.length ? `: ${consoleErrors.join(" | ")}` : ""}`);

  console.log("private window");
  const incognito = await chromium.launch({ channel: "chrome", headless: !process.argv.includes("--headed"), args: ["--enable-unsafe-webgpu"] });
  try {
    const p = await incognito.newPage();
    await open(p, false); // private windows get no service worker, so no isolation on Pages
    await waitForReady(p);
    const notice: string = await p.evaluate(() => (window as any).__wesper.notice);
    check(/didn't let the page keep .*encoder/.test(notice), `says the models will download again: "${notice}"`);
  } finally {
    await incognito.close();
  }
} catch (e) {
  failed = true;
  console.error(`FAIL: ${e instanceof Error ? e.message : e}`);
} finally {
  await browser.close();
  rmSync(profile, { recursive: true, force: true });
  stopServers();
  clearTimeout(watchdog);
}
console.log(failed ? "e2e failed" : "e2e passed");
process.exit(failed ? 1 : 0);
