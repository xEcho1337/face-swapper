import { defineConfig } from 'vite';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));

// Static site: web/ is the root, ../public is copied verbatim (models dir is
// gitignored / installed separately — never bundled with weights).
export default defineConfig({
  root: resolve(here, 'web'),
  base: './',
  publicDir: resolve(here, 'public'),
  build: {
    outDir: resolve(here, 'dist'),
    emptyOutDir: true,
    target: 'es2022',
    assetsInlineLimit: 0,
    sourcemap: true,
    rollupOptions: {
      // Multi-page: the Responsible Use policy is a real second page, so its
      // link never 404s on the static host.
      input: {
        index: resolve(here, 'web/index.html'),
        'responsible-use': resolve(here, 'web/responsible-use.html'),
      },
    },
  },
  worker: {
    format: 'es',
  },
  server: {
    port: 5173,
    headers: {
      // Needed so SharedArrayBuffer / multi-threaded WASM ORT works when the
      // user self-hosts with these headers. (Vite dev only; dist/ hosting
      // should set them too if threads are enabled.)
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
});
