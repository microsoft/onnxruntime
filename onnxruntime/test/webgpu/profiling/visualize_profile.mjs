// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import { readFile, mkdir, writeFile } from 'node:fs/promises';
import { basename, dirname, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { isDeepStrictEqual } from 'node:util';
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
    'WebGPU::Finish': 5,
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

// Sorted leaves retain report IDs. Subtree end times also find spans that start before a run.
export function createEventIndex(events) {
  const size = 2 ** Math.ceil(Math.log2(Math.max(1, events.length)));
  const ends = new Float64Array(size * 2).fill(-Infinity);
  for (let id = 0; id < events.length; id++) ends[size + id] = events[id][0] + events[id][1];
  for (let node = size - 1; node > 0; node--) ends[node] = Math.max(ends[node * 2], ends[node * 2 + 1]);
  return function* overlaps(start, stop, excludeEnd) {
    function* visit(node, first, last) {
      if (first >= events.length || ends[node] < start) return;
      const ts = events[first][0];
      if (ts > stop || (ts === stop && excludeEnd)) return;
      if (last - first === 1) {
        const overlap = Math.min(stop, ends[node]) - Math.max(start, ts);
        if (overlap > 0 || (overlap === 0 && ts >= start && (!excludeEnd || ts < stop))) yield first;
        return;
      }
      const mid = (first + last) / 2;
      yield* visit(node * 2, first, mid);
      yield* visit(node * 2 + 1, mid, last);
    }
    yield* visit(1, 0, size);
  };
}

function submissionLinks(events, strings) {
  const groups = new Map();
  for (let id = 0; id < events.length; id++) {
    const event = events[id];
    if (event[2] !== 6 && event[2] !== 10) continue;
    const key = JSON.parse(strings[event[4]]).submission_id;
    if (typeof key !== 'string' || !key) continue;
    const group = groups.get(key) ?? { id: key, submit: undefined, dispatches: [] };
    if (event[2] === 6) group.submit = group.submit === undefined ? id : null;
    else group.dispatches.push(id);
    groups.set(key, group);
  }
  // Missing or duplicate CPU submissions cannot establish an exact dependency.
  return [...groups.values()].filter((group) => Number.isInteger(group.submit) && group.dispatches.length);
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
    'submission_id',
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
  const overlaps = createEventIndex(compact);
  const summarize = (run, index, label = {}) => {
    const start = run.ts;
    const stop = start + run.dur;
    const excludeEnd = index !== -1 && run.dur > 0;
    const indices = [];
    const busy = [];
    const programs = new Map();
    const cpu = Array(lanes.length).fill(0);
    for (const id of overlaps(start, stop, excludeEnd)) {
      const event = compact[id];
      const overlap = Math.min(stop, event[0] + event[1]) - Math.max(start, event[0]);
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
    const busyUs = mergedDuration(busy);
    return {
      index,
      stage: String(label.stage ?? 'inference'),
      step: label.step ?? index,
      iteration: label.iteration,
      warmup: Boolean(label.warmup),
      wallMs: label.wall_ms,
      start,
      duration: run.dur,
      busy: busyUs,
      gap: Math.max(0, run.dur - busyUs),
      indices,
      cpu,
      programs: [...programs.values()].sort((a, b) => b.us - a.us),
    };
  };
  const summaries = runs.map((run, index) => summarize(run, index, captures[index] ?? {}));
  const traceStart = compact[0][0];
  const traceEnd = compact.reduce((end, event) => Math.max(end, event[0] + event[1]), traceStart);
  const wholeTrace = summarize({ ts: traceStart, dur: traceEnd - traceStart }, -1, { stage: 'whole trace' });
  if (Object.keys(baseline).length) {
    if (baseline.trace) throw new Error('Baseline must describe unprofiled measurements (trace must be null).');
    for (const key of ['model_sha256', 'ort_version', 'backend', 'workload']) {
      if (metadata[key] === undefined || baseline[key] !== metadata[key]) {
        throw new Error(`Baseline capture is missing or differs in ${key}.`);
      }
    }
    for (const key of ['capture_schema', 'inputs_sha256', 'session_options']) {
      if (
        (metadata[key] !== undefined || baseline[key] !== undefined) &&
        !isDeepStrictEqual(metadata[key], baseline[key])
      ) {
        throw new Error(`Baseline capture differs in ${key}.`);
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
  const submissions = submissionLinks(compact, strings);
  return {
    version: 1,
    title: parent || basename(modelPath),
    lanes,
    strings,
    events: compact,
    submissions,
    submissionCount: submissions.length,
    runs: summaries,
    wholeTrace,
    syntheticRun,
    alignment: [...new Set(gpu.map((event) => event.args.clock_alignment))].sort(),
    baseline: baselineMeans,
    gpuEvents: gpu.length,
    capture: {
      model: modelPath,
      modelSha256: metadata.model_sha256,
      ortVersion: metadata.ort_version,
      backend: metadata.backend,
      workload: metadata.workload,
      warmup: metadata.warmup,
      repeat: metadata.repeat,
      inputsSha256: metadata.inputs_sha256,
      sessionOptions: metadata.session_options,
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
  const annotations = await json(metadata);
  const report = buildReport(await json(tracePath), annotations, await json(baseline));
  if (!annotations.model) report.title = basename(tracePath);
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
    submissions: undefined,
    runs: report.runs.map(({ indices, ...run }) => run),
    wholeTrace: { ...report.wholeTrace, indices: undefined },
  };
  await writeFile(summaryPath, JSON.stringify(summary, null, 2));
  return { output: resolve(output), summary: resolve(summaryPath), report };
}

async function main() {
  const argv = process.argv.slice(2);
  if (!argv.length || argv.includes('--help')) {
    console.log(
      'node visualize_profile.mjs <ORT-trace.json> --output <report.html> [--metadata annotations.json] [--baseline baseline.json]',
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
