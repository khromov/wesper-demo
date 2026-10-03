// Model downloads with progress, kept in the Cache API so the ~1 GB arrives only once.
// The cache is an optimization: without it (private windows, quota, no Cache API) each visit
// downloads again.

const CACHE = "wesper-models-v1";

export type DownloadProgress = (loaded: number, total: number, fromCache: boolean) => void;

async function openCache(): Promise<Cache | undefined> {
  try {
    return await caches.open(CACHE);
  } catch {
    return undefined;
  }
}

export interface Downloaded {
  data: Uint8Array;
  /** The file is in the cache: it came from there, or was stored. */
  cached: boolean;
}

/** The file at url, which models.json says is `bytes` long. */
export async function download(url: string, bytes: number, onProgress: DownloadProgress): Promise<Downloaded> {
  const cache = await openCache();
  const hit = await cache?.match(url).catch(() => undefined);
  if (hit) {
    const buf = new Uint8Array(await hit.arrayBuffer());
    if (buf.length === bytes) {
      onProgress(bytes, bytes, true);
      return { data: buf, cached: true };
    }
  }

  const res = await fetch(url);
  if (!res.ok || !res.body) throw new Error(`${url}: HTTP ${res.status}`);
  const buf = new Uint8Array(bytes);
  const reader = res.body.getReader();
  let off = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (off + value.length > bytes) throw new Error(`${url} is larger than models.json says; reload the page`);
    buf.set(value, off);
    off += value.length;
    onProgress(off, bytes, false);
  }
  if (off !== bytes) throw new Error(`${url}: received ${off} of ${bytes} bytes`);

  if (!cache) return { data: buf, cached: false };
  try {
    await cache.put(url, new Response(buf));
    // Drop earlier exports of the same file: they differ only in the ?v= hash.
    const path = new URL(url).pathname;
    for (const req of await cache.keys()) if (req.url !== url && new URL(req.url).pathname === path) await cache.delete(req);
    return { data: buf, cached: true };
  } catch (e) {
    // e.g. over quota: private windows allow far less storage than the models need
    console.warn(`not caching ${url}:`, e);
    return { data: buf, cached: false };
  }
}

/** Deletes downloads whose URL isn't in `keep`. */
export async function pruneDownloads(keep: string[]): Promise<void> {
  const cache = await openCache();
  if (!cache) return;
  const wanted = new Set(keep);
  for (const req of await cache.keys()) if (!wanted.has(req.url)) await cache.delete(req);
}

/** Bytes this site stores, which is almost all downloaded models; null if the browser won't say. */
export async function storedBytes(): Promise<number | null> {
  try {
    return (await navigator.storage.estimate()).usage ?? null;
  } catch {
    return null;
  }
}

export async function clearDownloads(): Promise<void> {
  try {
    await caches.delete(CACHE);
  } catch {
    // no Cache API: nothing stored
  }
}
