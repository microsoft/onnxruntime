/**
 * Event-loop blocking test for InferenceSession.run().
 *
 * Measures the maximum event-loop lag while running squeezenet inference
 * concurrently with a high-frequency timer. A truly async run() keeps lag
 * near zero; a pseudo-async (setImmediate-wrapped sync) run() produces lag
 * equal to the inference wall-clock time.
 *
 * Usage:
 *   # against the local build (should show LOW lag — PASS):
 *   node test/test-event-loop-blocking.mjs
 *
 *   # against an installed onnxruntime-node package (should show HIGH lag — FAIL):
 *   node test/test-event-loop-blocking.mjs --pkg <path-to-onnxruntime-node-dir>
 *   e.g. --pkg C:/tmp/old/node_modules/onnxruntime-node
 */

import { readFileSync, existsSync } from 'fs';
import { createRequire } from 'module';
import { fileURLToPath } from 'url';
import { dirname, resolve } from 'path';
import { performance } from 'perf_hooks';

const __dirname = dirname(fileURLToPath(import.meta.url));

// Parse --pkg <dir> argument
const pkgIdx = process.argv.indexOf('--pkg');
const pkgDir = pkgIdx !== -1 ? resolve(process.argv[pkgIdx + 1]) : null;

let ort;
if (pkgDir) {
  if (!existsSync(pkgDir)) {
    console.error(`Package directory not found: ${pkgDir}`);
    process.exit(1);
  }
  const req = createRequire(pkgDir + '/package.json');
  ort = req(pkgDir);
  console.log(`Using package : ${pkgDir}`);
} else {
  ort = (await import('../dist/index.js')).default;
  console.log(`Using         : local build`);
}

console.log(`ORT version   : ${ort.env?.versions?.ortRelease ?? '(unknown)'}`);
console.log();

const modelPath = resolve(__dirname, 'testdata/squeezenet.onnx');
const inputData = JSON.parse(readFileSync(resolve(__dirname, 'testdata/squeezenet.input0.json'), 'utf8'));
const inputTensor = new ort.Tensor('float32', Float32Array.from(inputData), [1, 3, 224, 224]);

// --- event-loop lag monitor ---
const TICK_MS = 5;
let maxLagMs = 0;
let tickCount = 0;
let lastTick = performance.now();

const lagMonitor = setInterval(() => {
  const now = performance.now();
  const lag = now - lastTick - TICK_MS;
  if (lag > maxLagMs) maxLagMs = lag;
  lastTick = now;
  tickCount++;
}, TICK_MS);

// --- inference ---
const RUNS = 5;
console.log(`Loading squeezenet and running ${RUNS} concurrent inferences...`);
const session = await ort.InferenceSession.create(modelPath, { executionProviders: ['cpu'] });

const t0 = performance.now();
const promises = [];
for (let i = 0; i < RUNS; i++) {
  promises.push(session.run({ data: inputTensor }));
}
await Promise.all(promises);
const elapsed = performance.now() - t0;

clearInterval(lagMonitor);
await session.release();

// --- report ---
console.log(`Total inference time : ${elapsed.toFixed(0)} ms`);
console.log(`Timer ticks fired    : ${tickCount}`);
console.log(`Max event-loop lag   : ${maxLagMs.toFixed(1)} ms`);
console.log();

const PASS_THRESHOLD_MS = 50;
if (maxLagMs < PASS_THRESHOLD_MS) {
  console.log(`PASS — event loop remained responsive (lag < ${PASS_THRESHOLD_MS} ms)`);
} else {
  console.log(`FAIL — event loop was blocked for ${maxLagMs.toFixed(1)} ms`);
  process.exitCode = 1;
}
