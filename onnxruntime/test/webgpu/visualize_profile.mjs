// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import { readFile, mkdir, writeFile } from 'node:fs/promises';
import { basename, dirname, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { gzipSync } from 'node:zlib';

export const lanes = [
  ['Model run', 'CPU', '#64748b'],
  ['Graph execution', 'CPU', '#94a3b8'],
  ['Operator preparation', 'CPU', '#3b82f6'],
  ['Fallback operators', 'CPU', '#e8795a'],
  ['Tensor copies', 'CPU', '#06b6d4'],
  ['Command encoding', 'CPU', '#8b5cf6'],
  ['Queue submission', 'CPU', '#d94695'],
  ['Dawn waits', 'CPU', '#e5a331'],
  ['Profiling setup', 'CPU', '#a88545'],
  ['Profile readback', 'CPU', '#ea7945'],
  ['Compute dispatches', 'GPU', '#10b981'],
];

export function laneOf(event) {
  const args = event.args ?? {};
  if (typeof args.clock_alignment === 'string') return 10;
  const named = {
    model_run: 0,
    'SequentialExecutor::Execute': 1,
    'WebGPU::Encode': 5,
    'WebGPU::Submit': 6,
    'WebGPU::Wait': 7,
    'WebGPU::StartProfiling': 8,
    'WebGPU::ProfileReadback': 9,
  };
  if (Object.hasOwn(named, event.name)) return named[event.name];
  if (event.cat === 'Node' && event.name?.endsWith('_kernel_time')) {
    if (args.op_name?.startsWith('Memcpy')) return 4;
    return args.provider === 'CPUExecutionProvider' ? 3 : 2;
  }
  return -1;
}

export function mergedDuration(intervals) {
  let end = -Infinity;
  let total = 0;
  for (const [start, stop] of intervals.toSorted((a, b) => a[0] - b[0])) {
    total += Math.max(0, stop - Math.max(start, end));
    end = Math.max(end, stop);
  }
  return total;
}

// Keep measured timings authoritative: metadata annotates runs but cannot replace trace timestamps.
export function buildReport(trace, metadata = {}, baseline = {}) {
  const source = Array.isArray(trace) ? trace : trace?.traceEvents;
  if (!Array.isArray(source)) throw new Error('Expected an ORT event array or traceEvents array.');
  const events = source.filter(
    (event) => event.ph === 'X' && Number.isFinite(event.ts) && Number.isFinite(event.dur) && event.dur >= 0,
  );
  const gpu = events.filter((event) => laneOf(event) === 10);
  if (!gpu.length) throw new Error('No WebGPU GPU timestamps found. Enable profiling and timestamp-query support.');
  let runs = events.filter((event) => event.name === 'model_run').sort((a, b) => a.ts - b.ts);
  const syntheticRun = !runs.length;
  if (!runs.length) {
    const start = gpu.reduce((value, event) => Math.min(value, event.ts), Infinity);
    const end = gpu.reduce((value, event) => Math.max(value, event.ts + event.dur), start);
    runs = [{ ts: start, dur: end - start, name: 'GPU capture' }];
  }
  const captures = metadata.runs ?? [];
  if (!Array.isArray(captures)) throw new Error('Capture runs must be an array.');
  if (captures.length && captures.length !== runs.length)
    throw new Error('Capture run count does not match the trace.');
  const fields = [
    'op_name',
    'provider',
    'shapes',
    'cache_key',
    'input_type_shape',
    'output_type_shape',
    'clock_alignment',
    'execution_side',
  ];
  const strings = [];
  const interned = new Map();
  const intern = (value) => {
    if (!interned.has(value)) {
      interned.set(value, strings.length);
      strings.push(value);
    }
    return interned.get(value);
  };
  const compact = events
    .flatMap((event) => {
      const lane = laneOf(event);
      if (lane < 0) return [];
      const args = Object.fromEntries(
        fields.filter((key) => key in (event.args ?? {})).map((key) => [key, event.args[key]]),
      );
      return [
        [event.ts, event.dur, lane, intern(String(event.name)), intern(JSON.stringify(args)), event.pid, event.tid],
      ];
    })
    .sort((a, b) => a[0] - b[0]);
  const summaries = runs.map((run, index) => {
    const start = run.ts;
    const stop = start + run.dur;
    const indices = [];
    const busy = [];
    const programs = new Map();
    const cpu = Array(lanes.length).fill(0);
    for (let id = 0; id < compact.length; id++) {
      const event = compact[id];
      if (event[0] > stop || (event[0] === stop && run.dur > 0)) break;
      const overlap = Math.min(stop, event[0] + event[1]) - Math.max(start, event[0]);
      if (overlap < 0 || (overlap === 0 && (event[0] < start || (event[0] >= stop && run.dur > 0)))) continue;
      indices.push(id);
      if (event[2] === 10) {
        busy.push([Math.max(start, event[0]), Math.min(stop, event[0] + event[1])]);
        const program = strings[event[3]].split('&').at(-1);
        const item = programs.get(program) ?? { name: program, us: 0, count: 0 };
        item.us += overlap;
        item.count++;
        programs.set(program, item);
      } else {
        cpu[event[2]] += overlap;
      }
    }
    const label = captures[index] ?? {};
    const busyUs = mergedDuration(busy);
    return {
      index,
      stage: String(label.stage ?? 'inference'),
      step: label.step ?? index,
      iteration: label.iteration,
      warmup: Boolean(label.warmup),
      inputTokens: label.input_tokens,
      totalTokens: label.total_tokens,
      wallMs: label.wall_ms,
      start,
      duration: run.dur,
      busy: busyUs,
      gap: Math.max(0, run.dur - busyUs),
      indices,
      cpu,
      programs: [...programs.values()].sort((a, b) => b.us - a.us),
    };
  });
  if (Object.keys(baseline).length) {
    if (baseline.trace) throw new Error('Baseline must be captured with --no-profile.');
    for (const key of ['model_sha256', 'ort_version', 'backend', 'prompt_tokens', 'decode_steps', 'workload']) {
      if (metadata[key] === undefined || baseline[key] !== metadata[key]) {
        throw new Error(`Baseline capture is missing or differs in ${key}.`);
      }
    }
  }
  if (baseline.runs !== undefined && !Array.isArray(baseline.runs)) throw new Error('Baseline runs must be an array.');
  const baselineRuns = (baseline.runs ?? []).filter(
    (run) => !run.warmup && Number.isFinite(run.wall_ms) && run.wall_ms >= 0,
  );
  const baselineMeans = Object.create(null);
  for (const stage of new Set(baselineRuns.map((run) => run.stage))) {
    const selected = baselineRuns.filter((run) => run.stage === stage);
    baselineMeans[stage] = selected.reduce((sum, run) => sum + run.wall_ms, 0) / selected.length;
  }
  const modelPath = String(metadata.model ?? 'WebGPU capture').replaceAll('\\', '/');
  const parent = modelPath.split('/').slice(-3, -1).join(' / ');
  return {
    version: 1,
    title: parent || basename(modelPath),
    lanes,
    strings,
    events: compact,
    runs: summaries,
    syntheticRun,
    alignment: [...new Set(gpu.map((event) => event.args.clock_alignment))].sort(),
    baseline: baselineMeans,
    gpuEvents: gpu.length,
    capture: {
      model: modelPath,
      modelSha256: metadata.model_sha256,
      ortVersion: metadata.ort_version,
      backend: metadata.backend,
      promptTokens: metadata.prompt_tokens,
      workload: metadata.workload,
      warmup: metadata.warmup,
      repeat: metadata.repeat,
      decodeSteps: metadata.decode_steps,
    },
  };
}

export async function renderReport(tracePath, output, { metadata, baseline } = {}) {
  const summaryPath = output.replace(/\.html?$/i, '') + '.summary.json';
  const pathKey = (path) => (process.platform === 'win32' ? resolve(path).toLowerCase() : resolve(path));
  for (const source of [tracePath, metadata, baseline].filter(Boolean)) {
    if ([output, summaryPath].some((target) => pathKey(target) === pathKey(source))) {
      throw new Error('Output must not overwrite an input trace or capture.');
    }
  }
  if (pathKey(output) === pathKey(summaryPath)) throw new Error('HTML and summary paths must differ.');
  const json = async (path) => (path ? JSON.parse(await readFile(path, 'utf8')) : {});
  const report = buildReport(await json(tracePath), await json(metadata), await json(baseline));
  const root = dirname(fileURLToPath(import.meta.url));
  const template = await readFile(resolve(root, 'profile_report.html'), 'utf8');
  const client = await readFile(resolve(root, 'profile_report.js'), 'utf8');
  // Base64 contains no HTML delimiters. Trace strings are inserted into the DOM with textContent.
  const payload = gzipSync(JSON.stringify(report)).toString('base64');
  const page = template.replace('/* REPORT_CLIENT */', () => client).replace('REPORT_PAYLOAD', () => payload);
  await mkdir(dirname(resolve(output)), { recursive: true });
  await writeFile(output, page);
  const summary = {
    ...report,
    strings: undefined,
    events: undefined,
    runs: report.runs.map(({ indices, ...run }) => run),
  };
  await writeFile(summaryPath, JSON.stringify(summary, null, 2));
  return { output: resolve(output), summary: resolve(summaryPath), report };
}

async function main() {
  const argv = process.argv.slice(2);
  if (!argv.length || argv.includes('--help')) {
    console.log(
      'node visualize_profile.mjs <ORT-trace.json> --output <report.html> [--metadata capture.json] [--baseline capture.json]',
    );
    return;
  }
  const trace = argv.shift();
  const options = {};
  while (argv.length) {
    const key = argv.shift();
    if (!['--output', '--metadata', '--baseline'].includes(key) || !argv.length)
      throw new Error(`Unknown/missing option: ${key}`);
    options[key.slice(2)] = argv.shift();
  }
  if (!options.output) throw new Error('--output is required.');
  const result = await renderReport(trace, options.output, options);
  console.log(
    JSON.stringify(
      {
        output: result.output,
        summary: result.summary,
        runs: result.report.runs.length,
        gpuEvents: result.report.gpuEvents,
      },
      null,
      2,
    ),
  );
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  main().catch((error) => {
    console.error(error.message);
    process.exitCode = 1;
  });
}
