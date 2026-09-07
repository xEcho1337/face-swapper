/**
 * Persistent on-device weights cache (Cache Storage API).
 *
 * First visit downloads the weights (~730 MB total); every later visit loads
 * them from the visitor's own disk with zero network. Entries are keyed by
 * file and re-validated (size + sha256) on every read, so a corrupt or
 * outdated entry can never silently poison inference.
 *
 * Only REMOTE downloads are cached: same-host files in local dev are
 * mutable, so they are always read fresh (pass `key: null` to skip cache).
 *
 * Browser limits (quota-based, approximate — query the real numbers with
 * `navigator.storage.estimate()`):
 * - Chromium (Chrome/Edge/Brave, desktop): large pool, roughly up to ~60%
 *   of disk space shared across origins; `persist()` is often auto-granted
 *   on engaged sites; eviction is LRU under storage pressure.
 * - Firefox: percentage of free disk space (several GB in practice);
 *   `persist()` may prompt the user.
 * - Safari: most restrictive (~1 GB practical, aggressive eviction,
 *   especially without recent user interaction).
 * - Private/incognito windows: storage is ephemeral — cache won't survive.
 * 730 MB fits comfortably on normal desktops; phones/old Safari may evict.
 */
export const MODEL_CACHE_NAME = 'faceswapper-models-v1';

const cacheKey = (key) => `model:${key}`;

function cacheSupported() {
  try {
    return typeof caches !== 'undefined';
  } catch {
    return false;
  }
}

export async function sha256Hex(buf) {
  const digest = await crypto.subtle.digest('SHA-256', buf);
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, '0')).join('');
}

/**
 * Fetch a weights file, cache-first when `key` is set.
 * Returns `{ buf, source: 'cache' | 'network' }`.
 * `onProgress(receivedBytes, totalBytes|null)` fires while streaming.
 * Throws plain Errors with `.httpStatus` / `.isHtmlFallback` /
 * `.truncated` / `.badHash` flags for the caller to interpret.
 */
export async function cachedFetch(url, { key = null, expectedBytes = 0, sha256 = null, onProgress = null } = {}) {
  if (key && cacheSupported()) {
    try {
      const cache = await caches.open(MODEL_CACHE_NAME);
      const hit = await cache.match(cacheKey(key));
      if (hit) {
        const buf = await hit.arrayBuffer();
        const sizeOk = buf.byteLength >= expectedBytes;
        const hashOk = !sha256 || (await sha256Hex(buf)) === sha256;
        if (sizeOk && hashOk) return { buf, source: 'cache' };
        await cache.delete(cacheKey(key)); // stale/corrupt entry: refetch below
      }
    } catch {
      /* Cache API broken/unavailable (e.g. private mode) — use network */
    }
  }

  const res = await fetch(url);
  if (!res.ok) {
    const e = new Error(`HTTP ${res.status} for ${url}`);
    e.httpStatus = res.status;
    throw e;
  }
  const contentType = res.headers.get('content-type') || '';
  if (contentType.includes('text/html')) {
    const e = new Error(`server returned an HTML page instead of the file: ${url}`);
    e.isHtmlFallback = true;
    throw e;
  }
  const total = Number(res.headers.get('content-length')) || null;
  let buf;
  if (res.body && typeof onProgress === 'function') {
    const reader = res.body.getReader();
    const chunks = [];
    let received = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.byteLength;
      try {
        onProgress(received, total);
      } catch {
        /* progress must never break the download */
      }
    }
    const out = new Uint8Array(received);
    let offset = 0;
    for (const chunk of chunks) {
      out.set(chunk, offset);
      offset += chunk.byteLength;
    }
    buf = out.buffer;
  } else {
    buf = await res.arrayBuffer();
  }
  if (buf.byteLength < expectedBytes) {
    const e = new Error(`truncated file from ${url} (${buf.byteLength} bytes)`);
    e.truncated = true;
    throw e;
  }
  if (sha256 && (await sha256Hex(buf)) !== sha256) {
    const e = new Error(`sha256 mismatch for ${url} — not the verified reference copy`);
    e.badHash = true;
    throw e;
  }
  if (key && cacheSupported()) {
    try {
      const cache = await caches.open(MODEL_CACHE_NAME);
      await cache.put(
        cacheKey(key),
        new Response(buf.slice(0), { headers: { 'content-type': 'application/octet-stream' } }),
      );
    } catch {
      /* quota/private mode — run uncached */
    }
  }
  return { buf, source: 'network' };
}
