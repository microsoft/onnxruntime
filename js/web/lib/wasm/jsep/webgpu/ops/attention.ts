// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {TensorView} from '../../tensor-view';
import {createAttributeWithCacheKey} from '../attribute-with-cache-key';
import {ComputeContext, GpuDataType} from '../types';

import {castToF32, fillVector, getMaxComponents, inputVariable, outputVariable, ShaderHelper, sumVector, tensorTypeToWsglStorageType} from './common';

export const enum AttentionQkvFormat {
  unknown,          // enum value not set, or depends on qkv projection implementation details
  qkvBNSH,          // for non-packed qkv, permuted
  qkvBSNH,          // for non-packed qkv, not permuted, used by memory efficient attention or MultiHeadAttention
  qkvBSN3H,         // for TRT fused attention, qkv are packed
  qkvBNSHqkvBS3NH,  // for TRT fused causal attention, data has two formats (qkv is 3BNSH, gemm_buffer is BS3NH)
  qKvBSNHxBSN2H,    // for TRT fused cross attention, kv are packed
  qkvTNH,           // for memory efficient attention, qkv are not packed, and paddings are removed.
  qkvTN3H,          // for TRT fused attention, qkv are packed and paddings are removed
}

export const enum AttentionMaskType {
  none,                  // No mask
  mask1dKeySeqLen,       // [batch_size], key sequence length
  mask1dEndStart,        // [2 * batch_size] with end positions and start positions
  mask1DKeySeqLenStart,  // [3 * batch_size + 2] with [key_len[0], ..., key_len[batch_size - 1], query_start[0],
                         // ..., query_start[batch_size - 1], query_end[batch_size - 1], key_start[0], ...,
                         // key_start[batch_size - 1], key_end[batch_size - 1]]
  mask2dDummy,           // dummy mask with shape [1, 1] or [batch_size, 1]. It has same effect as no mask.
  mask2dKeyPadding,      // [batch_size, total_sequence_length]
  mask3dAttention,       // [batch_size, sequence_length, total_sequence_length]
  mask4dMegatron,        // Megatron causal mask with shape [batch_size, 1, max_sequence_length, max_sequence_length]
  maskUnknown
}

export interface AttentionParameters {
  batchSize: number;
  sequenceLength: number;
  pastSequenceLength: number;
  kvSequenceLength: number;
  totalSequenceLength: number;
  maxSequenceLength: number;
  inputHiddenSize: number;
  hiddenSize: number;
  vHiddenSize: number;
  headSize: number;
  vHeadSize: number;
  numHeads: number;
  isUnidirectional: boolean;
  pastPresentShareBuffer: boolean;
  maskFilterValue: number;
  maskType: AttentionMaskType;
  scale: number;
  broadcastResPosBias: boolean;
  passPastInKv: boolean;
  qkvFormat: AttentionQkvFormat;
}

export interface AttentionAttrs {
  numHeads: number;
  isUnidirectional: number;
  maskFilterValue: number;
  scale: number;
  doRotary: number;
  qkvHiddenSizes: number[];
  pastPresentShareBuffer: boolean;
}

const validateAttentionInputs = (inputs: readonly TensorView[], attributes: AttentionAttrs): AttentionParameters => {
  // Abbreviation and Meanings:
  //   B:    batch_size
  //   S:    sequence_length (input sequence length of query)
  //   P:    past_sequence_length (past sequence length of key or value)
  //   L:    kv_sequence_length (input sequence length of key or value)
  //   M:    max_sequence_length
  //   T:    total_sequence_length = past_sequence_length + kv_sequence_length
  //   N:    num_heads
  //   H:    head size for Q and K, aka q_head_size or k_head_size or qk_head_size
  //   H_v:  v_head_size
  //   D_i:  input hidden size
  //   D:    hidden size for Q and K (D = N * H), aka q_hidden_size or k_hidden_size or qk_hidden_size
  //   D_v:  v_hidden_size = num_heads * v_head_size

  // When past state is used, Q, K and V should have same hidden size (unless we split it into past_key and past_value).

  // Input shapes:
  //   input        (Q/K/V)    : (B, S, D_i)
  //   weights      (Q/K/V)    : (D_i, D + D + D_v)
  //   bias         (Q/K/V)    : (D + D + D_v)
  //   mask_index              : see below
  //   past         (K/V)      : (2, B, N, P, H) or NULL
  //   relative_position_bias            : (B, N, S, T) or NULL

  // For mask_index, the following shapes are supported:
  //     NULL, (B, 1), (1, 1)
  //     (B), (2 * B), (3 * B + 2)
  //     (B, T)
  //     (B, S, T)
  //     (B, 1, M, M)
  //
  // When a model is pruned (like some attention heads are removed in Q/K/V), input_hidden_size could be larger
  // than hidden dimension of Q, K and V.

  const input = inputs[0];
  const weights = inputs[1];
  const bias = inputs[2];
  const maskIndex = inputs[3];
  const past = inputs[4];
  const relativePositionBias = inputs[5];

  if (past && relativePositionBias) {
    throw new Error('Attention cannot have both past and relative_position_bias');
  }

  if (input.dims.length !== 3) {
    throw new Error('Input "input" must have 3 dimensions');
  }

  const batchSize = input.dims[0];
  const sequenceLength = input.dims[1];
  const inputHiddenSize = input.dims[2];

  if (bias.dims.length !== 1) {
    throw new Error('Input "bias" is expected to have 1 dimensions');
  }

  if (weights.dims.length !== 2) {
    throw new Error('Input "weights" is expected to have 2 dimensions');
  }

  if (weights.dims[0] !== inputHiddenSize) {
    throw new Error('Input 1 dimension 0 should have same length as dimension 2 of input 0');
  }

  if (bias.dims[0] !== weights.dims[1]) {
    throw new Error('Input "bias" dimension 0 should have same length as dimension 1 of input "weights"');
  }

  let qHiddenSize = bias.dims[0] / 3;
  let kHiddenSize = qHiddenSize;
  let vHiddenSize = kHiddenSize;
  if (attributes.qkvHiddenSizes.length > 0) {
    if (attributes.qkvHiddenSizes.length !== 3) {
      throw new Error('qkv_hidden_sizes attribute should have 3 elements');
    }
    for (const sz of attributes.qkvHiddenSizes) {
      if (sz % attributes.numHeads !== 0) {
        throw new Error('qkv_hidden_sizes should be divisible by num_heads');
      }
    }

    qHiddenSize = attributes.qkvHiddenSizes[0];
    kHiddenSize = attributes.qkvHiddenSizes[1];
    vHiddenSize = attributes.qkvHiddenSizes[2];
  }

  const kvSequenceLength = sequenceLength;

  if (qHiddenSize !== kHiddenSize) {
    throw new Error('qkv_hidden_sizes first element should be same as the second');
  }

  if (bias.dims[0] !== qHiddenSize + kHiddenSize + vHiddenSize) {
    throw new Error('Input "bias" dimension 0 should have same length as sum of Q/K/V hidden sizes');
  }

  let pastSequenceLength = 0;
  if (past) {
    if (kHiddenSize !== vHiddenSize) {
      throw new Error('Input "past" expect k_hidden_size == v_hidden_size');
    }
    if (past.dims.length !== 5) {
      throw new Error('Input "past" must have 5 dimensions');
    }
    if (past.dims[0] !== 2) {
      throw new Error('Input "past" first dimension must be 2');
    }
    if (past.dims[1] !== batchSize) {
      throw new Error('Input "past" second dimension must be batch_size');
    }
    if (past.dims[2] !== attributes.numHeads) {
      throw new Error('Input "past" third dimension must be num_heads');
    }
    if (past.dims[4] !== kHiddenSize / attributes.numHeads) {
      throw new Error('Input "past" fifth dimension must be k_hidden_size / num_heads');
    }

    if (!attributes.pastPresentShareBuffer) {
      pastSequenceLength = past.dims[3];
    }
    // TODO: handle past_seq_len
  }

  const totalSequenceLength = kvSequenceLength + pastSequenceLength;
  const maxSequenceLength = -1;

  const maskType = AttentionMaskType.none;
  if (maskIndex) {
    // maskType = AttentionMaskType.MASK_UNKNOWN;
    // TODO: handle mask
    throw new Error('Mask not supported');
  }

  if (past) {
    throw new Error('past is not supported');
  }
  if (relativePositionBias) {
    throw new Error('relativePositionBias is not supported');
  }

  return {
    batchSize,
    sequenceLength,
    pastSequenceLength,
    kvSequenceLength,
    totalSequenceLength,
    maxSequenceLength,
    inputHiddenSize,
    hiddenSize: qHiddenSize,
    vHiddenSize,
    headSize: Math.floor(qHiddenSize / attributes.numHeads),
    vHeadSize: Math.floor(vHiddenSize / attributes.numHeads),
    numHeads: attributes.numHeads,
    isUnidirectional: false,
    pastPresentShareBuffer: false,
    maskFilterValue: attributes.maskFilterValue,
    maskType,
    scale: attributes.scale,
    broadcastResPosBias: false,
    passPastInKv: false,
    qkvFormat: AttentionQkvFormat.qkvBNSH,
  };
};

export const parseAttentionAttributes = (attributes: AttentionAttrs): AttentionAttrs =>
    createAttributeWithCacheKey({...attributes});

export const computeInPlaceSoftmax = (context: ComputeContext, input: TensorView, n: number, d: number) => {
  const components = getMaxComponents(d);
  const inputHelper = outputVariable('x', input.dataType, input.dims, components);

  let threadMaxValue = 'threadMaxVector';
  if (components === 2) {
    threadMaxValue = 'max(threadMaxVector.x, threadMaxVector.y)';
  } else if (components === 4) {
    threadMaxValue = 'max(max(threadMaxVector.x, threadMaxVector.y), max(threadMaxVector.z, threadMaxVector.w))';
  }
  const dataType = tensorTypeToWsglStorageType(input.dataType);
  let WG = 64;
  const dComp = d / components;
  if (dComp < WG) {
    WG = 1;
  } else if (dComp / 8 < 64) {
    WG = Math.ceil(dComp / 8);
  }
  const elementsPerWG = Math.ceil(d / components / WG);

  const getShaderSource = (shaderHelper: ShaderHelper) => `
  const dInv: ${dataType} = 1 / ${d};
  const dComp = ${d / components};
  var<workgroup> wgMax: array<f32, ${WG}>;
  var<workgroup> wgSum: array<f32, ${WG}>;

  ${shaderHelper.declareVariables(inputHelper)}
  @compute @workgroup_size(${WG}, 1, 1)
  fn main(@builtin(workgroup_id) workgroup_id : vec3<u32>,
    @builtin(local_invocation_index) local_index : u32) {
    let localOffset = local_index * ${elementsPerWG};
    let offset: u32 = workgroup_id.x * dComp + localOffset;

    var threadMaxVector = ${fillVector('f32', components, '-3.402823e+38f')};
    for (var i: u32 = 0; i < ${elementsPerWG} && i + localOffset < dComp; i++) {
      threadMaxVector = max(${castToF32(dataType, components, 'x[offset + i]')}, threadMaxVector);
    }
    wgMax[local_index] = ${threadMaxValue};
    workgroupBarrier();

    var maxValue = -3.402823e+38f;
    for (var i = 0u; i < ${WG}; i++) {
      maxValue = max(wgMax[i], maxValue);
    }

    var sumVector = ${fillVector('f32', components, '0')};
    for (var i: u32 = 0; i < ${elementsPerWG} && i + localOffset < dComp; i++) {
      sumVector += exp(${castToF32(dataType, components, 'x[offset + i]')} - maxValue);
    }
    wgSum[local_index] = ${sumVector('sumVector', components)};
    workgroupBarrier();

    var sum: f32 = 0;
    for (var i = 0u; i < ${WG}; i++) {
      sum += wgSum[i];
    }

    if (sum == 0) {
      for (var i: u32 = 0; i < ${elementsPerWG} && i + localOffset < dComp; i++) {
        x[offset + i] = ${fillVector(dataType, components, 'dInv')};
      }
    } else {
      for (var i: u32 = 0; i < ${elementsPerWG} && i + localOffset < dComp; i++) {
        let f32input = ${castToF32(dataType, components, 'x[offset + i]')};
        x[offset + i] = ${inputHelper.type.value}(exp(f32input - maxValue) / sum);
      }
    }
  }`;

  context.compute(
      {
        name: 'AttentionProbsSoftmax',
        shaderCache: {hint: `${d}`},
        getShaderSource,
        getRunData: () => ({
          outputs: [],
          dispatchGroup: {x: n},
        }),
      },
      {inputs: [input], outputs: []});
};

const computeAttentionProbs =
    (context: ComputeContext, q: TensorView, key: TensorView, _bias: TensorView|undefined,
     parameters: AttentionParameters, attributes: AttentionAttrs) => {
      const probsShape = [
        parameters.batchSize, parameters.numHeads, parameters.sequenceLength,
        parameters.kvSequenceLength + parameters.pastSequenceLength
      ];
      // TODO: handle mask

      const alpha = attributes.scale === 0 ? 1.0 / Math.sqrt(parameters.headSize) : attributes.scale;

      const dataType = tensorTypeToWsglStorageType(q.dataType);

      const components = getMaxComponents(parameters.headSize);
      const qInput = inputVariable('q', q.dataType, q.dims, components);
      const kInput = inputVariable('key', key.dataType, key.dims, components);
      const output = outputVariable('output', q.dataType, probsShape);

      const vectorizedHeadSize = parameters.headSize / components;
      const M = parameters.sequenceLength;
      const N = parameters.totalSequenceLength;
      const K = vectorizedHeadSize;

      const TILE_SIZE = 12;

      const dispatch = {
        x: Math.ceil(parameters.totalSequenceLength / TILE_SIZE),
        y: Math.ceil(parameters.sequenceLength / TILE_SIZE),
        z: parameters.batchSize * parameters.numHeads
      };

      const inputs = [q, key];
      const getShaderSource = (shaderHelper: ShaderHelper) => `
  const M: u32 = ${M}u;
  const N: u32 = ${N}u;
  const K: u32 = ${K}u;
  const alpha: ${dataType} = ${alpha};
  const beta: ${dataType} = 1.0;
  const TILE_SIZE = ${TILE_SIZE}u;

  var<workgroup> tileQ: array<${qInput.type.storage}, ${TILE_SIZE * TILE_SIZE}>;
  var<workgroup> tileK: array<${qInput.type.storage}, ${TILE_SIZE * TILE_SIZE}>;

  ${shaderHelper.declareVariables(qInput, kInput, output)}

  @compute @workgroup_size(${TILE_SIZE}, ${TILE_SIZE}, 1)
  fn main(@builtin(workgroup_id) workgroup_id : vec3<u32>,
   @builtin(local_invocation_id) local_id : vec3<u32>, @builtin(local_invocation_index) local_index : u32) {
   let global_idx = (workgroup_id.z * ${dispatch.x * dispatch.y}u +
          workgroup_id.y * ${dispatch.x}u + workgroup_id.x) * ${TILE_SIZE * TILE_SIZE}u + local_index;

    // x holds the N and y holds the M
    let headIdx = workgroup_id.z;
    let m = workgroup_id.y * TILE_SIZE;
    let n = workgroup_id.x * TILE_SIZE;
    let lm = m + local_id.y;
    let ln = n + local_id.x;

    let qOffset = ${parameters.sequenceLength * vectorizedHeadSize} * headIdx + m * K;
    let kOffset = ${parameters.kvSequenceLength * vectorizedHeadSize} * headIdx + n * K;

    var value = ${fillVector(dataType, components)};
    for (var w: u32 = 0u; w < K; w += TILE_SIZE) {
      if (m + local_id.y < M && w + local_id.x < K) {
        tileQ[TILE_SIZE * local_id.y + local_id.x] = q[qOffset + local_id.y * K + w + local_id.x];
      }
      if (n + local_id.y < N && w + local_id.x < K) {
        tileK[TILE_SIZE * local_id.y + local_id.x] = key[kOffset + local_id.y * K + w + local_id.x];
      }
      workgroupBarrier();

      for (var k: u32 = 0u; k<TILE_SIZE && w+k < K; k++) {
        value += tileQ[TILE_SIZE * local_id.y + k] * tileK[TILE_SIZE * local_id.x + k];
      }

      workgroupBarrier();
    }

    let headOffset = headIdx * M * N;
    if (lm < M && ln < N) {
      let outputIdx = headOffset + lm * N + ln;
      output[outputIdx] = ${sumVector('value', components)} * alpha;
    }
  }`;

      const probs = context.compute(
          {
            name: 'AttentionProbs',
            shaderCache: {hint: JSON.stringify(parameters)},
            getRunData: () => ({
              outputs: [{dims: probsShape, dataType: q.dataType, gpuDataType: GpuDataType.default}],
              dispatchGroup: dispatch,
            }),
            getShaderSource,
          },
          {inputs, outputs: [-1]})[0];

      computeInPlaceSoftmax(
          context, probs, parameters.batchSize * parameters.numHeads * parameters.sequenceLength,
          parameters.totalSequenceLength);

      return probs;
    };

const computeVxAttentionScore =
    (context: ComputeContext, probs: TensorView, v: TensorView, params: AttentionParameters) => {
      const outputShape = [params.batchSize, params.sequenceLength, params.vHiddenSize];

      const probsHelper = inputVariable('probs', probs.dataType, probs.dims);
      const vHelper = inputVariable('v', v.dataType, v.dims);
      const output = outputVariable('output', probs.dataType, outputShape);

      const dataType = tensorTypeToWsglStorageType(probs.dataType);

      const TILE_SIZE = 12;
      const dispatch = {
        x: Math.ceil(params.vHeadSize / TILE_SIZE),
        y: Math.ceil(params.sequenceLength / TILE_SIZE),
        z: params.batchSize * params.numHeads
      };

      const getShaderSource = (shaderHelper: ShaderHelper) => `
  const M: u32 = ${params.sequenceLength}u;
  const N: u32 = ${params.vHeadSize}u;
  const K: u32 = ${params.totalSequenceLength}u;
  const numHeads: u32 = ${params.numHeads}u;
  const TILE_SIZE = ${TILE_SIZE}u;

  var<workgroup> tileQ: array<${probsHelper.type.storage}, ${TILE_SIZE * TILE_SIZE}>;
  var<workgroup> tileK: array<${probsHelper.type.storage}, ${TILE_SIZE * TILE_SIZE}>;

  ${shaderHelper.declareVariables(probsHelper, vHelper, output)}

  @compute @workgroup_size(${TILE_SIZE}, ${TILE_SIZE}, 1)
  fn main(@builtin(workgroup_id) workgroup_id : vec3<u32>,
   @builtin(local_invocation_id) local_id : vec3<u32>, @builtin(local_invocation_index) local_index : u32) {
   let global_idx = (workgroup_id.z * ${dispatch.x * dispatch.y}u +
          workgroup_id.y * ${dispatch.x}u + workgroup_id.x) * ${TILE_SIZE * TILE_SIZE}u + local_index;

   let headIdx = workgroup_id.z;
   let m = workgroup_id.y * TILE_SIZE + local_id.y;
   let n = workgroup_id.x * TILE_SIZE + local_id.x;

   let offsetA = headIdx * (M * K) + m * K;
   let offsetB = headIdx * (N * K) + n;

   var value = ${dataType}(0);
   for (var w: u32 = 0u; w < K; w += TILE_SIZE) {
     if (m < M && w + local_id.x < K) {
       tileQ[TILE_SIZE * local_id.y + local_id.x] = probs[offsetA + w + local_id.x];
     }
     if (n < N && w + local_id.y < K) {
       tileK[TILE_SIZE * local_id.y + local_id.x] = v[offsetB + (w + local_id.y) * N];
     }
     workgroupBarrier();
     for (var k: u32 = 0u; k<TILE_SIZE && w+k < K; k++) {
       value += tileQ[TILE_SIZE * local_id.y + k] * tileK[TILE_SIZE * k + local_id.x];
     }
     workgroupBarrier();
   }

   // we need to transpose output from BNSH_v to BSND_v
   let batchIdx = workgroup_id.z / ${params.numHeads};
   let currentBatchHeadNumber = workgroup_id.z % ${params.numHeads};
   let headOffset = (batchIdx * M * ${params.numHeads} + currentBatchHeadNumber) * ${params.vHeadSize};
   if (m < M && n < N) {
     let outputIdx = batchIdx * ${params.sequenceLength * params.vHiddenSize} + m * ${params.vHiddenSize}
       + currentBatchHeadNumber * ${params.vHeadSize} + n;
     output[outputIdx] = value;
   }
  }`;

      return context.compute(
          {
            name: 'AttentionScore',
            shaderCache: {hint: JSON.stringify(params)},
            getRunData: () => ({
              outputs: [{dims: outputShape, dataType: probs.dataType, gpuDataType: GpuDataType.default}],
              dispatchGroup: dispatch,
            }),
            getShaderSource,
          },
          {inputs: [probs, v], outputs: [0]})[0];
    };

export const applyAttention =
    (context: ComputeContext, q: TensorView, k: TensorView, v: TensorView, _maskIndex: TensorView|undefined,
     _past: TensorView|undefined, _pastKey: TensorView|undefined, _pastValue: TensorView|undefined,
     relativePositionBias: TensorView|undefined, parameters: AttentionParameters, attributes: AttentionAttrs) => {
      const probs = computeAttentionProbs(context, q, k, relativePositionBias, parameters, attributes);

      computeVxAttentionScore(context, probs, v, parameters);
    };

const prepare = (context: ComputeContext, parameters: AttentionParameters) => {
  const outputShape = [
    parameters.batchSize,
    parameters.numHeads,
    parameters.sequenceLength,
    parameters.headSize,
  ];

  const dataType = tensorTypeToWsglStorageType(context.inputs[0].dataType);

  const M = parameters.sequenceLength;
  const K = parameters.inputHiddenSize;
  const N = parameters.headSize;

  const TILE_SIZE = 12;
  const dispatch = {
    x: Math.ceil(parameters.headSize / TILE_SIZE),
    y: Math.ceil(parameters.sequenceLength / TILE_SIZE),
    z: parameters.batchSize * parameters.numHeads
  };

  const getShaderSource = () => `
  const M: u32 = ${M}u;
  const K: u32 = ${K}u;
  const N: u32 = ${N}u;
  const numHeads: u32 = ${parameters.numHeads};
  const ldb = ${parameters.hiddenSize + parameters.hiddenSize + parameters.vHiddenSize}u;
  const TILE_SIZE = ${TILE_SIZE}u;

  var<workgroup> tileInput: array<${dataType}, ${TILE_SIZE * TILE_SIZE}>;
  var<workgroup> tileWeightQ: array<${dataType}, ${TILE_SIZE * TILE_SIZE}>;
  var<workgroup> tileWeightK: array<${dataType}, ${TILE_SIZE * TILE_SIZE}>;
  var<workgroup> tileWeightV: array<${dataType}, ${TILE_SIZE * TILE_SIZE}>;

  @group(0) @binding(0) var<storage, read> input: array<${dataType}>;
  @group(0) @binding(1) var<storage, read> weight: array<${dataType}>;
  @group(0) @binding(2) var<storage, read> bias: array<${dataType}>;
  @group(0) @binding(3) var<storage, read_write> outputQ: array<${dataType}>;
  @group(0) @binding(4) var<storage, read_write> outputK: array<${dataType}>;
  @group(0) @binding(5) var<storage, read_write> outputV: array<${dataType}>;

  @compute @workgroup_size(${TILE_SIZE}, ${TILE_SIZE}, 1)
  fn main(@builtin(workgroup_id) workgroup_id : vec3<u32>,
   @builtin(local_invocation_id) local_id : vec3<u32>, @builtin(local_invocation_index) local_index : u32) {
   let global_idx = (workgroup_id.z * ${dispatch.x * dispatch.y}u +
          workgroup_id.y * ${dispatch.x}u + workgroup_id.x) * ${TILE_SIZE * TILE_SIZE}u + local_index;

    let batchIndex = workgroup_id.z / ${parameters.numHeads};
    let headNumber = workgroup_id.z % ${parameters.numHeads};
    let m = workgroup_id.y * TILE_SIZE + local_id.y;
    let n = workgroup_id.x * TILE_SIZE + local_id.x;

    let inputOffset = batchIndex * (M * K) + m * K;
    let biasOffsetQ = headNumber * ${parameters.headSize};
    let biasOffsetK = ${parameters.hiddenSize} + biasOffsetQ;
    let biasOffsetV = ${parameters.hiddenSize} + biasOffsetK;

    var valueQ = ${dataType}(0);
    var valueK = ${dataType}(0);
    var valueV = ${dataType}(0);
    for (var w: u32 = 0u; w < K; w += TILE_SIZE) {
      if (m < M && w + local_id.x < K) {
        tileInput[TILE_SIZE * local_id.y + local_id.x] = input[inputOffset + w + local_id.x];
      }
      if (n < N && w + local_id.y < K) {
        let offset = n + (w + local_id.y) * ldb;
        tileWeightQ[TILE_SIZE * local_id.y + local_id.x] = weight[biasOffsetQ + offset];
        tileWeightK[TILE_SIZE * local_id.y + local_id.x] = weight[biasOffsetK + offset];
        tileWeightV[TILE_SIZE * local_id.y + local_id.x] = weight[biasOffsetV + offset];
      }
      workgroupBarrier();
      for (var k: u32 = 0u; k<TILE_SIZE && w+k < K; k++) {
        let inputTileOffset = TILE_SIZE * local_id.y + k;
        let weightTileOffset = TILE_SIZE * k + local_id.x;
        valueQ += tileInput[inputTileOffset] * tileWeightQ[weightTileOffset];
        valueK += tileInput[inputTileOffset] * tileWeightK[weightTileOffset];
        valueV += tileInput[inputTileOffset] * tileWeightV[weightTileOffset];
      }

      workgroupBarrier();
    }

    let headOffset = (m * N + n) % ${parameters.headSize};
    valueQ += bias[headOffset + biasOffsetQ];
    valueK += bias[headOffset + biasOffsetK];
    valueV += bias[headOffset + biasOffsetV];

    let offset = workgroup_id.z * M * N;
    if (m < M && n < N) {
      let outputIdx = offset + m * N + n;
      outputQ[outputIdx] = valueQ;
      outputK[outputIdx] = valueK;
      outputV[outputIdx] = valueV;
    }
  }`;

  const inputs = [context.inputs[0], context.inputs[1], context.inputs[2]];

  return context.compute(
      {
        name: 'AttentionPrepare',
        shaderCache: {hint: JSON.stringify(parameters)},
        getRunData: () => ({
          outputs: [
            {dims: outputShape, dataType: context.inputs[0].dataType, gpuDataType: GpuDataType.default},
            {dims: outputShape, dataType: context.inputs[0].dataType, gpuDataType: GpuDataType.default},
            {dims: outputShape, dataType: context.inputs[0].dataType, gpuDataType: GpuDataType.default},
          ],
          dispatchGroup: dispatch,
        }),
        getShaderSource,
      },
      {inputs, outputs: [-1, -1, -1]});
};

export const attention = (context: ComputeContext, attributes: AttentionAttrs): void => {
  const params = validateAttentionInputs(context.inputs, attributes);

  const [q, k, v] = prepare(context, params);

  return applyAttention(
      context, q, k, v, context.inputs[4], undefined, undefined, undefined, context.inputs[5], params, attributes);
};
