// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import { DataType } from '../../../wasm-common';
import { TensorView } from '../../tensor-view';
import { ShapeUtil } from '../../util';
import { AttributeWithCacheKey, createAttributeWithCacheKey } from '../attribute-with-cache-key';
import { ComputeContext, ProgramInfo, ProgramUniform } from '../types';

import { inputVariable, outputVariable, ShaderHelper, tensorTypeToWsglValueType } from './common';

// ONNX DFT (opset 17-20): a 1-D transform along `axis`, with the innermost dimension holding the real/imaginary
// parts (1 for real input, 2 for complex). Forward is unnormalized, inverse is scaled by 1/N — matching the CPU
// kernel core/providers/cpu/signal/dft.cc. 5-smooth lengths use a shared-memory mixed-radix (2/3/4/5) Stockham
// FFT (one transform per workgroup); other lengths fall back to a direct O(N^2) DFT (the CPU kernel uses
// Bluestein chirp-z there, which would restore O(N log N) if the fallback ever becomes hot).

export interface DftAttributes extends AttributeWithCacheKey {
  readonly axis: number;
  readonly inverse: number;
  readonly onesided: number;
}

const WORKGROUP_SIZE = 256;
const MAX_SHARED_MEMORY_LENGTH = 512;
const TWO_PI = 2 * Math.PI;

const factorizeToRadices = (length: number): number[] | undefined => {
  const radices: number[] = [];
  let remaining = length;
  for (const radix of [4, 2, 3, 5]) {
    while (remaining % radix === 0) {
      radices.push(radix);
      remaining /= radix;
    }
  }
  return remaining === 1 ? radices : undefined;
};

const wgslFloat = (value: number): string => {
  const text = value.toPrecision(9);
  return /[.eE]/.test(text) ? text : `${text}.0`;
};

// One radix-`radix` Stockham stage: reads the `subTransform`-sized partial transforms from `readOffset` and
// writes the combined transforms to the other half of the ping-pong buffer, with twiddles baked in.
const stageCode = (radix: number, subTransform: number, length: number, readOffset: number, sign: number): string => {
  const butterflies = length / radix;
  const writeOffset = MAX_SHARED_MEMORY_LENGTH - readOffset;
  const out = (k: number): string => `smem[${writeOffset}u + base + ${k * subTransform}u]`;
  let code = `  for (var t = local_idx; t < ${butterflies}u; t += ${WORKGROUP_SIZE}u) {\n`;
  code += `    let twiddleIndex = t % ${subTransform}u;\n    let angleUnit = f32(twiddleIndex);\n`;
  code += `    var leg: array<vec2<f32>, 5>;\n`;
  for (let j = 0; j < radix; j++) {
    const source = `${readOffset}u + t + ${j * butterflies}u`;
    if (j === 0) {
      code += `    leg[0] = smem[${source}];\n`;
    } else {
      const twiddle = (sign * TWO_PI * j) / (radix * subTransform);
      code += `    { let a = ${wgslFloat(twiddle)} * angleUnit; leg[${j}] = cmul(smem[${source}], vec2<f32>(cos(a), sin(a))); }\n`;
    }
  }
  code += `    let base = (t / ${subTransform}u) * ${subTransform * radix}u + twiddleIndex;\n`;
  if (radix === 2) {
    code += `    ${out(0)} = leg[0] + leg[1];\n    ${out(1)} = leg[0] - leg[1];\n`;
  } else if (radix === 4) {
    const rotate = sign < 0 ? 'vec2<f32>(oddDiff.y, -oddDiff.x)' : 'vec2<f32>(-oddDiff.y, oddDiff.x)';
    code += `    let evenSum = leg[0] + leg[2]; let evenDiff = leg[0] - leg[2];\n`;
    code += `    let oddSum = leg[1] + leg[3]; let oddDiff = leg[1] - leg[3];\n`;
    code += `    let oddRot = ${rotate};\n`;
    code += `    ${out(0)} = evenSum + oddSum;\n    ${out(1)} = evenDiff + oddRot;\n`;
    code += `    ${out(2)} = evenSum - oddSum;\n    ${out(3)} = evenDiff - oddRot;\n`;
  } else {
    for (let k = 0; k < radix; k++) {
      const terms = ['leg[0]'];
      for (let j = 1; j < radix; j++) {
        const angle = (sign * TWO_PI * (j * k)) / radix;
        const c = wgslFloat(Math.cos(angle));
        const s = wgslFloat(Math.sin(angle));
        terms.push(`vec2<f32>(leg[${j}].x*${c} - leg[${j}].y*${s}, leg[${j}].x*${s} + leg[${j}].y*${c})`);
      }
      code += `    ${out(k)} = ${terms.join(' + ')};\n`;
    }
  }
  return `${code}  }\n  workgroupBarrier();\n`;
};

const fftStages = (radices: number[], length: number, sign: number): { code: string; resultOffset: number } => {
  let code = '';
  let subTransform = 1;
  let readOffset = 0;
  for (const radix of radices) {
    code += stageCode(radix, subTransform, length, readOffset, sign);
    subTransform *= radix;
    readOffset = MAX_SHARED_MEMORY_LENGTH - readOffset;
  }
  return { code, resultOffset: readOffset };
};

interface DftPlan {
  dataType: number;
  outputDims: number[];
  length: number; // transform length N, after any dft_length / IRFFT resize
  signalLength: number;
  inner: number; // transforms packed between the axis and the innermost (complex) dimension
  batch: number;
  inputComponents: number;
  outputComponents: number;
  outputLength: number;
  inverse: boolean;
  onesided: boolean;
}

const computeDftPlan = (
  input: TensorView,
  fftAxis: number,
  inverse: boolean,
  onesided: boolean,
  dftLength: number | undefined,
): DftPlan => {
  const dims = input.dims;
  const rank = dims.length;
  const inputComponents = dims[rank - 1];
  const signalLength = dims[fftAxis];
  let length = inverse && onesided ? (signalLength - 1) * 2 : signalLength;
  if (dftLength !== undefined) {
    length = dftLength;
  }
  const outputComponents = inverse && onesided ? 1 : 2;
  const outputLength = onesided && !inverse ? Math.floor(length / 2) + 1 : length;
  const outputDims = dims.slice();
  outputDims[fftAxis] = outputLength;
  outputDims[rank - 1] = outputComponents;
  let inner = 1;
  for (let d = fftAxis + 1; d < rank - 1; d++) {
    inner *= dims[d];
  }
  const batch = ShapeUtil.size(dims) / inputComponents / signalLength;
  return {
    dataType: input.dataType,
    outputDims,
    length,
    signalLength,
    inner,
    batch,
    inputComponents,
    outputComponents,
    outputLength,
    inverse,
    onesided,
  };
};

// Only the values that change the emitted code go into the cache key; the addressing dimensions
// (signalLength/inner/outputLength/batch) are passed as uniforms so shaders are reused across shapes.
const programHint = (plan: DftPlan, path: string): string =>
  [path, plan.length, plan.inputComponents, plan.outputComponents, plan.inverse, plan.onesided].join(';');

const programUniforms = (plan: DftPlan): ProgramUniform[] => [
  { type: DataType.uint32, data: plan.batch },
  { type: DataType.uint32, data: plan.signalLength },
  { type: DataType.uint32, data: plan.inner },
  { type: DataType.uint32, data: plan.outputLength },
];

const declareDftUniforms = (
  shaderHelper: ShaderHelper,
  x: ReturnType<typeof inputVariable>,
  y: ReturnType<typeof outputVariable>,
): string =>
  shaderHelper
    .registerUniform('batch', 'u32')
    .registerUniform('signalLength', 'u32')
    .registerUniform('inner', 'u32')
    .registerUniform('outputLength', 'u32')
    .declareVariables(x, y);

const createFftProgramInfo = (plan: DftPlan): ProgramInfo => {
  const { dataType, length, inputComponents, outputComponents, inverse, onesided } = plan;
  const valueType = tensorTypeToWsglValueType(dataType);
  const sign = inverse ? 1 : -1;
  const scale = inverse ? 1 / length : 1;
  const radices = factorizeToRadices(length)!;

  const getShaderSource = (shaderHelper: ShaderHelper): string => {
    const x = inputVariable('x', dataType, [1]);
    const y = outputVariable('y', dataType, [1]);
    const readSample = (indexExpr: string): string => {
      const offset = `inBase + (${indexExpr}) * uniforms.inner * ${inputComponents}u`;
      const real = `f32(${x.getByOffset(offset)})`;
      const imag = inputComponents === 2 ? `f32(${x.getByOffset(`${offset} + 1u`)})` : '0.0';
      return `vec2<f32>(${real}, ${imag})`;
    };

    let load: string;
    if (inverse && onesided) {
      const conjugateEnd = length % 2 === 0 ? 'uniforms.signalLength - 1u' : 'uniforms.signalLength';
      load = `
    for (var i = local_idx; i < uniforms.signalLength; i += ${WORKGROUP_SIZE}u) {
      smem[i] = ${readSample('i')};
    }
    workgroupBarrier();
    for (var k = local_idx + 1u; k < ${conjugateEnd}; k += ${WORKGROUP_SIZE}u) {
      let h = smem[k];
      smem[${length}u - k] = vec2<f32>(h.x, -h.y);
    }
    workgroupBarrier();`;
    } else {
      load = `
    let loadCount = min(uniforms.signalLength, ${length}u);
    for (var i = local_idx; i < ${length}u; i += ${WORKGROUP_SIZE}u) {
      if (i < loadCount) { smem[i] = ${readSample('i')}; } else { smem[i] = vec2<f32>(0.0); }
    }
    workgroupBarrier();`;
    }

    const { code: stages, resultOffset } = fftStages(radices, length, sign);
    const scaled = scale === 1 ? `smem[${resultOffset}u + i]` : `smem[${resultOffset}u + i] * ${wgslFloat(scale)}`;
    const storeImag = outputComponents === 2 ? y.setByOffset('off + 1u', `${valueType}(v.y)`) : '';

    return `
  ${declareDftUniforms(shaderHelper, x, y)}
  var<workgroup> smem: array<vec2<f32>, ${2 * MAX_SHARED_MEMORY_LENGTH}>;
  fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
  }
  ${shaderHelper.mainStart(WORKGROUP_SIZE)}
    let row = workgroup_index;
    if (row >= uniforms.batch) { return; }
    let outer = row / uniforms.inner;
    let within = row % uniforms.inner;
    let inBase = (outer * uniforms.signalLength * uniforms.inner + within) * ${inputComponents}u;
    let outBase = (outer * uniforms.outputLength * uniforms.inner + within) * ${outputComponents}u;
    ${load}
${stages}    for (var i = local_idx; i < uniforms.outputLength; i += ${WORKGROUP_SIZE}u) {
      let v = ${scaled};
      let off = outBase + i * uniforms.inner * ${outputComponents}u;
      ${y.setByOffset('off', `${valueType}(v.x)`)}
      ${storeImag}
    }
  }`;
  };

  return {
    name: 'DFT',
    shaderCache: { hint: programHint(plan, 'fft'), inputDependencies: ['type'] },
    getShaderSource,
    getRunData: () => ({
      outputs: [{ dims: plan.outputDims, dataType }],
      programUniforms: programUniforms(plan),
      dispatchGroup: { x: plan.batch },
    }),
  };
};

// Direct O(N^2) DFT for lengths the shared-memory FFT cannot take (non 5-smooth, or beyond the workgroup
// budget). One workgroup per transform; each output bin sums over the input samples read from global memory.
const createDirectDftProgramInfo = (plan: DftPlan): ProgramInfo => {
  const { dataType, length, inputComponents, outputComponents, inverse, onesided } = plan;
  const valueType = tensorTypeToWsglValueType(dataType);
  const sign = inverse ? 1 : -1;
  const scale = inverse ? 1 / length : 1;

  const getShaderSource = (shaderHelper: ShaderHelper): string => {
    const x = inputVariable('x', dataType, [1]);
    const y = outputVariable('y', dataType, [1]);
    const readSample = (indexExpr: string): string => {
      const offset = `inBase + (${indexExpr}) * uniforms.inner * ${inputComponents}u`;
      const real = `f32(${x.getByOffset(offset)})`;
      const imag = inputComponents === 2 ? `f32(${x.getByOffset(`${offset} + 1u`)})` : '0.0';
      return `vec2<f32>(${real}, ${imag})`;
    };

    // For IRFFT the spectrum is the Hermitian extension of the half-spectrum input.
    const spectrum =
      inverse && onesided
        ? `fn spectrum(inBase: u32, k: u32) -> vec2<f32> {
    if (k < uniforms.signalLength) { return ${readSample('k')}; }
    let h = ${readSample(`${length}u - k`)};
    return vec2<f32>(h.x, -h.y);
  }`
        : `fn spectrum(inBase: u32, n: u32) -> vec2<f32> {
    if (n < uniforms.signalLength) { return ${readSample('n')}; }
    return vec2<f32>(0.0, 0.0);
  }`;

    // knMod tracks (k*n) mod length via addition, so the twiddle index never overflows u32 at large N.
    const accumulate = `
      let angle = ${wgslFloat(sign * TWO_PI)} * f32(knMod) / ${wgslFloat(length)};
      acc += cmul(spectrum(inBase, n), vec2<f32>(cos(angle), sin(angle)));
      knMod += k;
      if (knMod >= ${length}u) { knMod -= ${length}u; }`;
    const storeImag = outputComponents === 2 ? y.setByOffset('off + 1u', `${valueType}(v.y)`) : '';
    const scaled = scale === 1 ? 'acc' : `acc * ${wgslFloat(scale)}`;

    return `
  ${declareDftUniforms(shaderHelper, x, y)}
  fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
  }
  ${spectrum}
  ${shaderHelper.mainStart(WORKGROUP_SIZE)}
    let row = workgroup_index;
    if (row >= uniforms.batch) { return; }
    let outer = row / uniforms.inner;
    let within = row % uniforms.inner;
    let inBase = (outer * uniforms.signalLength * uniforms.inner + within) * ${inputComponents}u;
    let outBase = (outer * uniforms.outputLength * uniforms.inner + within) * ${outputComponents}u;
    for (var k = local_idx; k < uniforms.outputLength; k += ${WORKGROUP_SIZE}u) {
      var acc = vec2<f32>(0.0, 0.0);
      var knMod = 0u;
      for (var n = 0u; n < ${length}u; n++) {${accumulate}
      }
      let v = ${scaled};
      let off = outBase + k * uniforms.inner * ${outputComponents}u;
      ${y.setByOffset('off', `${valueType}(v.x)`)}
      ${storeImag}
    }
  }`;
  };

  return {
    name: 'DFT',
    shaderCache: { hint: programHint(plan, 'direct'), inputDependencies: ['type'] },
    getShaderSource,
    getRunData: () => ({
      outputs: [{ dims: plan.outputDims, dataType }],
      programUniforms: programUniforms(plan),
      dispatchGroup: { x: plan.batch },
    }),
  };
};

const readScalar = (tensor: TensorView | undefined): number | undefined => {
  if (!tensor || ShapeUtil.size(tensor.dims) === 0) {
    return undefined;
  }
  return tensor.dataType === DataType.int32 ? tensor.getInt32Array()[0] : Number(tensor.getBigInt64Array()[0]);
};

const validateInputs = (inputs: readonly TensorView[]): void => {
  if (!inputs || inputs.length < 1) {
    throw new Error('DFT requires at least 1 input.');
  }
  const complex = inputs[0].dims[inputs[0].dims.length - 1];
  if (complex !== 1 && complex !== 2) {
    throw new Error("DFT input's innermost dimension must be 1 (real) or 2 (complex).");
  }
};

export const dft = (context: ComputeContext, attributes: DftAttributes): void => {
  validateInputs(context.inputs);
  const input = context.inputs[0];
  const rank = input.dims.length;
  const inverse = attributes.inverse !== 0;
  const onesided = attributes.onesided !== 0;

  // opset 20 passes dft_length (input 1) and axis (input 2) as scalar tensors; opset 17-19 use the axis
  // attribute. Both are read on the CPU, so only the signal is uploaded to the GPU.
  const dftLength = readScalar(context.inputs[1]);
  const fftAxis = ShapeUtil.normalizeAxis(readScalar(context.inputs[2]) ?? attributes.axis, rank);
  if (fftAxis === rank - 1) {
    throw new Error('DFT axis must refer to a signal dimension, not the innermost (real/imaginary) dimension.');
  }
  if (inverse && onesided && input.dims[rank - 1] !== 2) {
    throw new Error('Inverse one-sided DFT (IRFFT) requires complex-valued input (innermost dimension 2).');
  }

  const plan = computeDftPlan(input, fftAxis, inverse, onesided, dftLength);
  const useSharedMemoryFft = plan.length <= MAX_SHARED_MEMORY_LENGTH && factorizeToRadices(plan.length) !== undefined;
  const programInfo = useSharedMemoryFft ? createFftProgramInfo(plan) : createDirectDftProgramInfo(plan);
  context.compute(programInfo, { inputs: [0] });
};

export const parseDftAttributes = (attributes: Record<string, unknown>): DftAttributes =>
  createAttributeWithCacheKey({
    axis: (attributes.axis as number) ?? 1,
    inverse: (attributes.inverse as number) ?? 0,
    onesided: (attributes.onesided as number) ?? 0,
  });
