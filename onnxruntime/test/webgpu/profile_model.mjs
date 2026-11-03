// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Generic fixed-input native WebGPU capture. Requires a WebGPU-enabled onnxruntime-node build.
import { createHash } from 'node:crypto';
import { mkdir, readFile, readdir, writeFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { dirname, resolve } from 'node:path';
import { performance } from 'node:perf_hooks';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { parseArgs } from 'node:util';

const sha256 = (data) => createHash('sha256').update(data).digest('hex');
const types = {
  float32: Float32Array,
  float64: Float64Array,
  float16: Uint16Array, // Inline values are the raw IEEE 754 half-precision bit patterns.
  int8: Int8Array,
  uint8: Uint8Array,
  int16: Int16Array,
  uint16: Uint16Array,
  int32: Int32Array,
  uint32: Uint32Array,
  int64: BigInt64Array,
  uint64: BigUint64Array,
  bool: Uint8Array,
};

export function tensorData(spec, bytes) {
  const ArrayType = Object.hasOwn(types, spec.type) ? types[spec.type] : undefined;
  if (!ArrayType) throw new Error(`Unsupported tensor type: ${spec.type}`);
  if (!Array.isArray(spec.dims) || !spec.dims.every((dim) => Number.isSafeInteger(dim) && dim >= 0))
    throw new Error('Tensor dims must be nonnegative safe integers.');
  const count = spec.dims.reduce((size, dim) => {
    const product = size * dim;
    if (!Number.isSafeInteger(product)) throw new Error('Tensor element count exceeds the safe integer range.');
    return product;
  }, 1);
  const size = count * ArrayType.BYTES_PER_ELEMENT;
  if (!Number.isSafeInteger(size)) throw new Error('Tensor byte length exceeds the safe integer range.');
  if (bytes !== undefined) {
    if (spec.data !== undefined) throw new Error('Specify data or data_file, not both.');
    if (bytes.byteLength !== size) throw new Error('Tensor data_file byte length does not match dims/type.');
    // Own an aligned copy; input files contain little-endian numeric data.
    return new ArrayType(Uint8Array.from(bytes).buffer);
  }
  if (!Array.isArray(spec.data) || spec.data.length !== count)
    throw new Error('Tensor data length does not match dims.');
  const big = spec.type === 'int64' || spec.type === 'uint64';
  const data = new ArrayType(count);
  for (let index = 0; index < count; index++) {
    const value = spec.data[index];
    if (big) {
      if (!(typeof value === 'string' && /^-?\d+$/.test(value)) && !Number.isSafeInteger(value))
        throw new Error('64-bit integers must be safe numbers or decimal strings.');
      data[index] = BigInt(value);
      if (data[index] !== BigInt(value)) throw new Error('Integer value is outside the tensor type range.');
    } else {
      if (typeof value !== 'number' || !Number.isFinite(value))
        throw new Error('Tensor values must be finite numbers.');
      data[index] = value;
      if (!Number.isFinite(data[index])) throw new Error('Value overflows the tensor type.');
      if (!spec.type.startsWith('float') || spec.type === 'float16') {
        if (data[index] !== value) throw new Error('Integer value is outside the tensor type range.');
      }
      if (spec.type === 'bool' && value !== 0 && value !== 1) throw new Error('Boolean values must be 0 or 1.');
    }
  }
  return data;
}

export async function loadInputs(path, Tensor) {
  if (new Uint8Array(new Uint16Array([1]).buffer)[0] !== 1) throw new Error('Little-endian hosts are required.');
  const specs = JSON.parse(await readFile(path, 'utf8'));
  if (!specs || Array.isArray(specs) || typeof specs !== 'object' || !Object.keys(specs).length)
    throw new Error('Inputs must be a nonempty object mapping input names to tensor specifications.');
  const feeds = Object.create(null);
  const hash = createHash('sha256');
  try {
    for (const name of Object.keys(specs).sort()) {
      const spec = specs[name];
      if (!spec || typeof spec !== 'object') throw new Error(`Invalid input specification: ${name}`);
      if (spec.data_file !== undefined && typeof spec.data_file !== 'string')
        throw new Error('data_file must be a path.');
      const bytes = spec.data_file === undefined ? undefined : await readFile(resolve(dirname(path), spec.data_file));
      const data = tensorData(spec, bytes);
      hash.update(JSON.stringify([name, spec.type, spec.dims]));
      hash.update(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
      feeds[name] = new Tensor(spec.type, data, spec.dims);
    }
  } catch (error) {
    for (const tensor of Object.values(feeds)) tensor.dispose();
    throw error;
  }
  return { feeds, inputsSha256: hash.digest('hex') };
}

export function summarizeOutputs(outputs) {
  return Object.fromEntries(
    Object.entries(outputs).map(([name, tensor]) => {
      const source = tensor.data;
      if (!ArrayBuffer.isView(source)) throw new Error('Only numeric tensor outputs are supported.');
      const data =
        tensor.type === 'float16' ? new Uint16Array(source.buffer, source.byteOffset, source.byteLength / 2) : source;
      const finite =
        tensor.type === 'float16'
          ? !data.some((value) => (value & 0x7c00) === 0x7c00)
          : !data.some((value) => typeof value === 'number' && !Number.isFinite(value));
      if (!finite) throw new Error(`Non-finite output: ${name}`);
      return [
        name,
        {
          type: tensor.type,
          dims: tensor.dims,
          sha256: sha256(new Uint8Array(data.buffer, data.byteOffset, data.byteLength)),
          sample: Array.from(data.subarray(0, 8), (value) => (typeof value === 'bigint' ? String(value) : value)),
        },
      ];
    }),
  );
}

export function validateOptions(options) {
  if (!['D3D12', 'Vulkan', 'Metal'].includes(options.backend)) throw new Error('Unsupported WebGPU backend.');
  for (const [name, minimum] of [
    ['warmup', 0],
    ['repeat', 1],
    ['intraOpThreads', 0],
  ]) {
    if (!Number.isSafeInteger(options[name]) || options[name] < minimum)
      throw new Error(`${name} must be an integer >= ${minimum}.`);
  }
}

export async function capture(options, ort) {
  validateOptions(options);
  if (!ort.listSupportedBackends?.().some((backend) => backend.name === 'webgpu' && backend.bundled))
    throw new Error('Use an onnxruntime-node build with native WebGPU enabled.');
  const model = resolve(options.model);
  const modelSha256 = sha256(await readFile(model));
  const { feeds, inputsSha256 } = await loadInputs(resolve(options.inputs), ort.Tensor);
  let session;
  try {
    // A fresh directory avoids overwrites and makes the single generated trace unambiguous.
    const output = resolve(options.outputDir);
    await mkdir(dirname(output), { recursive: true });
    await mkdir(output);
    const sessionOptions = {
      executionProviders: ['webgpu'],
      intraOpNumThreads: options.intraOpThreads,
      enableProfiling: !options.noProfile,
      logSeverityLevel: 3,
      extra: {
        'session.disable_cpu_ep_fallback': '1',
        'ep.webgpuexecutionprovider.dawnBackendType': options.backend,
      },
    };
    session = await ort.InferenceSession.create(model, {
      ...sessionOptions,
      ...(!options.noProfile ? { profileFilePrefix: resolve(output, 'webgpu') } : {}),
    });
    if (JSON.stringify([...session.inputNames].sort()) !== JSON.stringify(Object.keys(feeds).sort()))
      throw new Error('Specify every model input exactly once.');
    const runs = [];
    let outputSummary;
    for (let iteration = 0; iteration < options.warmup + options.repeat; iteration++) {
      const start = performance.now();
      const outputs = await session.run(feeds);
      const elapsed = performance.now() - start;
      try {
        // Validation/hashing happens outside the measured interval, after the first measured run.
        if (iteration === options.warmup) outputSummary = summarizeOutputs(outputs);
      } finally {
        for (const tensor of Object.values(outputs)) tensor.dispose();
      }
      runs.push({ iteration, warmup: iteration < options.warmup, stage: 'inference', wall_ms: elapsed });
    }
    let trace = null;
    if (!options.noProfile) {
      session.endProfiling();
      const traces = (await readdir(output)).filter((name) => name.startsWith('webgpu_') && name.endsWith('.json'));
      if (traces.length !== 1) throw new Error('Expected exactly one ORT profile trace.');
      trace = resolve(output, traces[0]);
    }
    const metadata = {
      capture_schema: 2,
      model,
      model_sha256: modelSha256,
      inputs_sha256: inputsSha256,
      ort_version: ort.env.versions.node,
      ort_module: resolve(options.ortModule),
      backend: options.backend,
      session_options: sessionOptions,
      warmup: options.warmup,
      repeat: options.repeat,
      trace,
      runs,
      outputs: outputSummary,
      workload: 'Repeated fixed-input inference; host tensors and outputs; graph capture disabled',
      timing: 'wall_ms includes Node scheduling, inference and CPU output transfer; excludes validation/disposal',
      command: process.argv,
    };
    // enableProfiling is the sole intentional difference from a separately captured baseline.
    delete metadata.session_options.enableProfiling;
    await writeFile(resolve(output, 'capture.json'), JSON.stringify(metadata, null, 2));
    return metadata;
  } finally {
    try {
      if (session) await session.release();
    } finally {
      for (const tensor of Object.values(feeds)) tensor.dispose();
    }
  }
}

async function main() {
  const { values, positionals } = parseArgs({
    allowPositionals: true,
    options: {
      inputs: { type: 'string' },
      'output-dir': { type: 'string' },
      'ort-module': { type: 'string', default: fileURLToPath(new URL('../../../js/node', import.meta.url)) },
      backend: { type: 'string', default: 'D3D12' },
      warmup: { type: 'string', default: '1' },
      repeat: { type: 'string', default: '3' },
      'intra-op-threads': { type: 'string', default: '1' },
      'no-profile': { type: 'boolean', default: false },
      help: { type: 'boolean' },
    },
  });
  if (values.help) {
    console.log(
      'node profile_model.mjs <model.onnx> --inputs <inputs.json> --output-dir <new-directory> [--ort-module <native-node-package>] [--no-profile] [--warmup 1] [--repeat 3] [--backend D3D12]',
    );
    return;
  }
  if (positionals.length !== 1 || !values.inputs || !values['output-dir'])
    throw new Error('Specify one model, --inputs and --output-dir; use --help for options.');
  const options = {
    model: positionals[0],
    inputs: values.inputs,
    outputDir: values['output-dir'],
    ortModule: values['ort-module'],
    backend: values.backend,
    warmup: Number(values.warmup),
    repeat: Number(values.repeat),
    intraOpThreads: Number(values['intra-op-threads']),
    noProfile: values['no-profile'],
  };
  validateOptions(options);
  const ort = createRequire(import.meta.url)(resolve(options.ortModule));
  const metadata = await capture(options, ort);
  console.log(
    JSON.stringify({ trace: metadata.trace, runs: metadata.runs.length, outputs: metadata.outputs }, null, 2),
  );
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) await main();
