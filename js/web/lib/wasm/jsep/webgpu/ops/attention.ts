// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import { DataType } from '../../../wasm-common';
import { TensorView } from '../../tensor-view';
import { ShapeUtil } from '../../util';
import { ComputeContext, GpuDataType, ProgramInputTensorInfoDependency, ProgramUniform } from '../types';

import {
  getMaxComponents,
  inputVariable,
  outputVariable,
  ShaderHelper,
  tensorTypeToWsglStorageType,
  tensorTypeToWsglValueType,
  UniformDataElementType,
  UniformsArrayType,
} from './common';

export const enum AttentionQkvFormat {
  unknown, // enum value not set, or depends on qkv projection implementation details
  qkvBNSH, // for non-packed qkv, permuted
  qkvBSNH, // for non-packed qkv, not permuted, used by memory efficient attention or MultiHeadAttention
  qkvBSN3H, // for TRT fused attention, qkv are packed
  qkvBNSHqkvBS3NH, // for TRT fused causal attention, data has two formats (qkv is 3BNSH, gemm_buffer is BS3NH)
  qKvBSNHxBSN2H, // for TRT fused cross attention, kv are packed
  qkvTNH, // for memory efficient attention, qkv are not packed, and paddings are removed.
  qkvTN3H, // for TRT fused attention, qkv are packed and paddings are removed
}

export const enum AttentionMaskType {
  none, // No mask
  mask1dKeySeqLen, // [batch_size], key sequence length
  mask1dEndStart, // [2 * batch_size] with end positions and start positions
  mask1DKeySeqLenStart, // [3 * batch_size + 2] with [key_len[0], ..., key_len[batch_size - 1], query_start[0],
  // ..., query_start[batch_size - 1], query_end[batch_size - 1], key_start[0], ...,
  // key_start[batch_size - 1], key_end[batch_size - 1]]
  mask2dDummy, // dummy mask with shape [1, 1] or [batch_size, 1]. It has same effect as no mask.
  mask2dKeyPadding, // [batch_size, total_sequence_length]
  mask3dAttention, // [batch_size, sequence_length, total_sequence_length]
  mask4dMegatron, // Megatron causal mask with shape [batch_size, 1, max_sequence_length, max_sequence_length]
  maskUnknown,
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
  kvNumHeads?: number;
  nReps?: number;
  isUnidirectional?: boolean;
  pastPresentShareBuffer: boolean;
  maskFilterValue?: number;
  maskType: AttentionMaskType;
  scale: number;
  broadcastResPosBias: boolean;
  passPastInKv: boolean;
  qkvFormat: AttentionQkvFormat;
  isPastkvBSNH?: boolean;
}

export interface AttentionAttrs {
  numHeads: number;
  kvNumHeads?: number;
  isUnidirectional?: number;
  maskFilterValue?: number;
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
  //   attention_bias          : (B, N, S, T) or NULL

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
  const attentionBias = inputs[5];

  if (past && attentionBias) {
    throw new Error('Attention cannot have both past and attention_bias');
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

  if (attentionBias) {
    if (attentionBias.dims.length !== 4) {
      throw new Error('Input "attention_bias" must have 4 dimensions');
    }

    // TODO: support broadcasting the first and second dimensions of attention_bias
    if (
      attentionBias.dims[0] !== batchSize ||
      attentionBias.dims[1] !== attributes.numHeads ||
      attentionBias.dims[2] !== sequenceLength ||
      attentionBias.dims[3] !== totalSequenceLength
    ) {
      throw new Error('Expect "attention_bias" shape (batch_size, num_heads, sequence_length, total_sequence_length)');
    }
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

const createInPlaceSoftmaxProgramInfo = (input: TensorView, n: number, d: number) => {
  const components = getMaxComponents(d);
  let WG = 64;
  const dComp = d / components;
  if (dComp < WG) {
    WG = 32;
  }
  const elementsPerThread = Math.ceil(d / components / WG);
  const programUniforms: ProgramUniform[] = [
    { type: DataType.float, data: 1 / d },
    { type: DataType.uint32, data: dComp },
    { type: DataType.uint32, data: elementsPerThread },
  ];
  const dataType = tensorTypeToWsglStorageType(input.dataType, components);
  const f32Type = tensorTypeToWsglValueType(DataType.float, components);
  const inputDependencies: ProgramInputTensorInfoDependency[] = ['type'];
  const getShaderSource = (shaderHelper: ShaderHelper) => {
    const inputHelper = outputVariable('x', input.dataType, input.dims, components);
    const elemValueType = tensorTypeToWsglValueType(input.dataType);
    const uniforms: UniformsArrayType = [
      { name: 'd_inv', type: 'f32' },
      { name: 'd_comp', type: 'u32' },
      { name: 'elements_per_thread', type: 'u32' },
    ];

    return `
  var<workgroup> thread_max: array<f32, ${WG}>;
  var<workgroup> thread_sum: array<f32, ${WG}>;
  ${shaderHelper.registerUniforms(uniforms).declareVariables(inputHelper)}
  ${shaderHelper.mainStart([WG, 1, 1])}
    let local_offset = local_idx * uniforms.elements_per_thread;
    let offset = (global_idx / ${WG}) * uniforms.d_comp + local_offset;

    var thread_max_vector = ${f32Type}(-3.402823e+38f);
    for (var i: u32 = 0; i < uniforms.elements_per_thread && i + local_offset < uniforms.d_comp; i++) {
      thread_max_vector = max(${f32Type}(x[offset + i]), thread_max_vector);
    }
    thread_max[local_idx] = ${(() => {
      switch (components) {
        case 1:
          return 'thread_max_vector';
        case 2:
          return 'max(thread_max_vector.x, thread_max_vector.y)';
        case 4:
          return 'max(max(thread_max_vector.x, thread_max_vector.y), max(thread_max_vector.z, thread_max_vector.w))';
        default:
          throw new Error(`Unsupported components: ${components}`);
      }
    })()};
    workgroupBarrier();

    var max_value =  f32(-3.402823e+38f);
    for (var i = 0u; i < ${WG}; i++) {
      max_value = max(thread_max[i], max_value);
    }

    var sum_vector = ${f32Type}(0);
    for (var i: u32 = 0; i < uniforms.elements_per_thread && i + local_offset < uniforms.d_comp; i++) {
      sum_vector += exp(${f32Type}(x[offset + i]) - max_value);
    }
    thread_sum[local_idx] = ${(() => {
      switch (components) {
        case 1:
          return 'sum_vector';
        case 2:
          return 'sum_vector.x + sum_vector.y';
        case 4:
          return 'sum_vector.x + sum_vector.y + sum_vector.z + sum_vector.w';
        default:
          throw new Error(`Unsupported components: ${components}`);
      }
    })()};
    workgroupBarrier();

    var sum: f32 = 0;
    for (var i = 0u; i < ${WG}; i++) {
      sum += thread_sum[i];
    }

    if (sum == 0) {
      for (var i: u32 = 0; i < uniforms.elements_per_thread && i + local_offset < uniforms.d_comp; i++) {
        x[offset + i] = ${inputHelper.type.value}(${elemValueType}(uniforms.d_inv));
      }
    } else {
      for (var i: u32 = 0; i < uniforms.elements_per_thread && i + local_offset < uniforms.d_comp; i++) {
        var f32input = ${f32Type}(x[offset + i]);
        x[offset + i] = ${inputHelper.type.value}(exp(f32input - max_value) / sum);
      }
    }
  }`;
  };

  return {
    name: 'AttentionProbsSoftmax',
    shaderCache: { hint: `${WG};${dataType};${components}`, inputDependencies },
    getShaderSource,
    getRunData: () => ({ outputs: [], dispatchGroup: { x: n }, programUniforms }),
  };
};

const createAttentionProbsProgramInfo = (
  outputCount: number,
  q: TensorView,
  key: TensorView,
  pastKey: TensorView | undefined,
  attentionBias: TensorView | undefined,
  parameters: AttentionParameters,
  attributes: AttentionAttrs,
  pastSequenceLength: number,
) => {
  const totalSequenceLength = pastSequenceLength + parameters.kvSequenceLength;
  const probsShape = [parameters.batchSize, parameters.numHeads, parameters.sequenceLength, totalSequenceLength];
  const presentKey = parameters.kvNumHeads === undefined && outputCount > 1 && pastKey;
  const presentKeyShape = presentKey
    ? [parameters.batchSize, parameters.numHeads, totalSequenceLength, parameters.headSize]
    : undefined;

  // TODO: handle mask

  const alpha = attributes.scale === 0 ? 1.0 / Math.sqrt(parameters.headSize) : attributes.scale;
  const components = getMaxComponents(parameters.headSize);
  const vectorizedHeadSize = parameters.headSize / components;
  const TILE_SIZE = 12;
  const dispatch = {
    x: Math.ceil(totalSequenceLength / TILE_SIZE),
    y: Math.ceil(parameters.sequenceLength / TILE_SIZE),
    z: parameters.batchSize * parameters.numHeads,
  };
  const programUniforms: ProgramUniform[] = [
    { type: DataType.uint32, data: parameters.sequenceLength },
    { type: DataType.uint32, data: vectorizedHeadSize },
    { type: DataType.uint32, data: totalSequenceLength },
    { type: DataType.uint32, data: parameters.numHeads },
    { type: DataType.float, data: alpha },
    { type: DataType.uint32, data: pastSequenceLength },
    { type: DataType.uint32, data: parameters.kvSequenceLength },
  ];
  // Feed pastKey to the shader-code only if it is non-zero and presentKey is being produced
  const feedPastKey = presentKey && pastKey && ShapeUtil.size(pastKey.dims) > 0;
  const inputDependencies: ProgramInputTensorInfoDependency[] = ['type', 'type'];
  if (feedPastKey) {
    inputDependencies.push('type');
  }
  if (attentionBias) {
    inputDependencies.push('type');
  }
  const outputs = [{ dims: probsShape, dataType: q.dataType, gpuDataType: GpuDataType.default }];
  if (presentKey) {
    outputs.push({ dims: presentKeyShape!, dataType: q.dataType, gpuDataType: GpuDataType.default });
  }
  const getShaderSource = (shaderHelper: ShaderHelper) => {
    const qInput = inputVariable('q', q.dataType, q.dims, components);
    const kInput = inputVariable('key', key.dataType, key.dims, components);
    const inputVars = [qInput, kInput];
    if (feedPastKey) {
      const pastKeyInput = inputVariable('past_key', pastKey.dataType, pastKey.dims, components);
      inputVars.push(pastKeyInput);
    }
    if (attentionBias) {
      inputVars.push(inputVariable('attention_bias', attentionBias.dataType, attentionBias.dims));
    }
    const output = outputVariable('output', q.dataType, probsShape);
    const outputVars = [output];
    if (presentKey) {
      outputVars.push(outputVariable('present_key', q.dataType, presentKeyShape!, components));
    }
    const f32Type = tensorTypeToWsglValueType(DataType.float, components);

    const uniforms: UniformsArrayType = [
      { name: 'M', type: 'u32' },
      { name: 'K', type: 'u32' },
      { name: 'N', type: 'u32' },
      { name: 'num_heads', type: 'u32' },
      { name: 'alpha', type: 'f32' as UniformDataElementType },
      { name: 'past_sequence_length', type: 'u32' },
      { name: 'kv_sequence_length', type: 'u32' },
    ];
    return `
  const TILE_SIZE = ${TILE_SIZE}u;

  var<workgroup> tileQ: array<${qInput.type.storage}, ${TILE_SIZE * TILE_SIZE}>;
  var<workgroup> tileK: array<${qInput.type.storage}, ${TILE_SIZE * TILE_SIZE}>;
  ${shaderHelper.registerUniforms(uniforms).declareVariables(...inputVars, ...outputVars)}
  ${shaderHelper.mainStart([TILE_SIZE, TILE_SIZE, 1])}
    // x holds the N and y holds the M
    let headIdx = workgroup_id.z;
    let m = workgroup_id.y * TILE_SIZE;
    let n = workgroup_id.x * TILE_SIZE;
    let qOffset = uniforms.M * uniforms.K * headIdx + m * uniforms.K;
    ${(() => {
      if (feedPastKey && presentKey) {
        return `
    let kOffset = uniforms.kv_sequence_length * uniforms.K * headIdx;
    let pastKeyOffset = uniforms.past_sequence_length * uniforms.K * headIdx;`;
      } else {
        return `
    let kOffset = uniforms.N * uniforms.K * headIdx + n * uniforms.K;`;
      }
    })()}
    ${presentKey ? 'let presentKeyOffset = headIdx * uniforms.N * uniforms.K;' : ''}
    var value = ${f32Type}(0);
    for (var w: u32 = 0u; w < uniforms.K; w += TILE_SIZE) {
      if (global_id.y < uniforms.M && w + local_id.x < uniforms.K) {
        tileQ[TILE_SIZE * local_id.y + local_id.x] = q[qOffset + local_id.y * uniforms.K + w + local_id.x];
      }
      if (n + local_id.y < uniforms.N && w + local_id.x < uniforms.K) {
        var idx = TILE_SIZE * local_id.y + local_id.x;
      ${(() => {
        if (feedPastKey && presentKey) {
          return `
              if (n + local_id.y < uniforms.past_sequence_length) {
                tileK[idx] = past_key[pastKeyOffset + (n + local_id.y) * uniforms.K + w + local_id.x];
              } else {
                tileK[idx] =
                         key[kOffset + (n + local_id.y - uniforms.past_sequence_length) * uniforms.K + w + local_id.x];
              }`;
        } else {
          return 'tileK[idx] = key[kOffset + local_id.y * uniforms.K + w + local_id.x];';
        }
      })()}
      ${
        presentKey ? 'present_key[presentKeyOffset + (n + local_id.y) * uniforms.K + w + local_id.x] = tileK[idx];' : ''
      }
      }
      workgroupBarrier();

      for (var k: u32 = 0u; k < TILE_SIZE && w+k < uniforms.K; k++) {
        value += ${f32Type}(tileQ[TILE_SIZE * local_id.y + k] * tileK[TILE_SIZE * local_id.x + k]);
      }

      workgroupBarrier();
    }

    let headOffset = headIdx * uniforms.M * uniforms.N;
    if (global_id.y < uniforms.M && global_id.x < uniforms.N) {
      let outputIdx = headOffset + global_id.y * uniforms.N + global_id.x;
      var sum: f32 = ${(() => {
        switch (components) {
          case 1:
            return 'value';
          case 2:
            return 'value.x + value.y';
          case 4:
            return 'value.x + value.y + value.z + value.w';
          default:
            throw new Error(`Unsupported components: ${components}`);
        }
      })()};
        output[outputIdx] = ${output.type.value} (sum * uniforms.alpha) + ${
          attentionBias ? 'attention_bias[outputIdx]' : '0.0'
        };
    }
  }`;
  };
  return {
    name: 'AttentionProbs',
    shaderCache: {
      hint: `${components};${attentionBias !== undefined};${pastKey !== undefined};${outputCount}`,
      inputDependencies,
    },
    getRunData: () => ({ outputs, dispatchGroup: dispatch, programUniforms }),
    getShaderSource,
  };
};

const createVxAttentionScoreProgramInfo = (
  outputCount: number,
  probs: TensorView,
  v: TensorView,
  pastValue: TensorView | undefined,
  params: AttentionParameters,
  pastSequenceLength: number,
) => {
  const totalSequenceLength = pastSequenceLength + params.kvSequenceLength;
  const nReps = params.nReps ? params.nReps : 1;
  const repeatedVHiddenSize = params.vHiddenSize * nReps;
  const presentValue = params.kvNumHeads == null && outputCount > 1 && pastValue;
  const presentValueShape = presentValue
    ? [params.batchSize, params.numHeads, totalSequenceLength, params.headSize]
    : undefined;
  const outputShape = [params.batchSize, params.sequenceLength, repeatedVHiddenSize];
  const TILE_SIZE = 12;
  const dispatch = {
    x: Math.ceil(params.vHeadSize / TILE_SIZE),
    y: Math.ceil(params.sequenceLength / TILE_SIZE),
    z: params.batchSize * params.numHeads,
  };

  const programUniforms: ProgramUniform[] = [
    { type: DataType.uint32, data: params.sequenceLength },
    { type: DataType.uint32, data: totalSequenceLength },
    { type: DataType.uint32, data: params.vHeadSize },
    { type: DataType.uint32, data: params.numHeads },
    { type: DataType.uint32, data: repeatedVHiddenSize },
    { type: DataType.uint32, data: pastSequenceLength },
    { type: DataType.uint32, data: params.kvSequenceLength },
  ];
  // Feed pastValue to the shader-code only if it is non-empty and presentValue is being produced
  const feedPastValue = presentValue && pastValue && ShapeUtil.size(pastValue.dims) > 0;
  const inputDependencies: ProgramInputTensorInfoDependency[] = ['type', 'type'];
  if (feedPastValue) {
    inputDependencies.push('type');
  }
  const outputs = [{ dims: outputShape, dataType: probs.dataType, gpuDataType: GpuDataType.default }];
  if (presentValue) {
    outputs.push({ dims: presentValueShape!, dataType: probs.dataType, gpuDataType: GpuDataType.default });
  }
  const getShaderSource = (shaderHelper: ShaderHelper) => {
    const probsHelper = inputVariable('probs', probs.dataType, probs.dims);
    const vHelper = inputVariable('v', v.dataType, v.dims);
    const inputVars = [probsHelper, vHelper];
    if (feedPastValue) {
      inputVars.push(inputVariable('past_value', pastValue.dataType, pastValue.dims));
    }
    const output = outputVariable('output', probs.dataType, outputShape);
    const outputVars = [output];
    if (presentValue) {
      outputVars.push(outputVariable('present_value', probs.dataType, presentValueShape!));
    }
    const uniforms: UniformsArrayType = [
      { name: 'M', type: 'u32' },
      { name: 'K', type: 'u32' },
      { name: 'N', type: 'u32' },
      { name: 'num_heads', type: 'u32' },
      { name: 'v_hidden_size', type: 'u32' },
      { name: 'past_sequence_length', type: 'u32' },
      { name: 'kv_sequence_length', type: 'u32' },
    ];
    return `
  const TILE_SIZE = ${TILE_SIZE}u;
  var<workgroup> tileQ: array<${probsHelper.type.value}, ${TILE_SIZE * TILE_SIZE}>;
  var<workgroup> tileK: array<${probsHelper.type.value}, ${TILE_SIZE * TILE_SIZE}>;
  ${shaderHelper.registerUniforms(uniforms).declareVariables(...inputVars, ...outputVars)}
  ${shaderHelper.mainStart([TILE_SIZE, TILE_SIZE, 1])}
   let headIdx = workgroup_id.z;
   let m = global_id.y;
   let n = global_id.x;

   let offsetA = headIdx * (uniforms.M * uniforms.K) + m * uniforms.K;
   ${(() => {
     if (feedPastValue && presentValue) {
       return `
    let pastValueOffset = headIdx * uniforms.N * uniforms.past_sequence_length + n;
    let vOffset = headIdx * uniforms.N * uniforms.kv_sequence_length + n;
      `;
     } else {
       return `
   let offsetB = headIdx * uniforms.N * uniforms.K + n;
            `;
     }
   })()}
    ${presentValue ? 'let presentValueOffset = headIdx * uniforms.N * uniforms.K + n;' : ''}
   var value = ${probsHelper.type.storage}(0);
   for (var w: u32 = 0u; w < uniforms.K; w += TILE_SIZE) {
      if (m < uniforms.M && w + local_id.x < uniforms.K) {
        tileQ[TILE_SIZE * local_id.y + local_id.x] = probs[offsetA + w + local_id.x];
      }
      if (n < uniforms.N && w + local_id.y < uniforms.K) {
        var idx = TILE_SIZE * local_id.y + local_id.x;
        ${(() => {
          if (feedPastValue && presentValue) {
            return `
        if (w + local_id.y < uniforms.past_sequence_length) {
          tileK[idx] = past_value[pastValueOffset + (w + local_id.y) * uniforms.N];
        } else {
          tileK[idx] = v[vOffset + (w + local_id.y - uniforms.past_sequence_length) * uniforms.N];
        }
      `;
          } else {
            return `
        tileK[idx] = v[offsetB + (w + local_id.y) * uniforms.N];
      `;
          }
        })()}
        ${presentValue ? 'present_value[presentValueOffset + (w + local_id.y) * uniforms.N] = tileK[idx];' : ''}
      }
     workgroupBarrier();
     for (var k: u32 = 0u; k < TILE_SIZE && w+k < uniforms.K; k++) {
       value += tileQ[TILE_SIZE * local_id.y + k] * tileK[TILE_SIZE * k + local_id.x];
     }
     workgroupBarrier();
   }

   // we need to transpose output from BNSH_v to BSND_v
   let batchIdx = workgroup_id.z / uniforms.num_heads;
   let currentBatchHeadNumber = workgroup_id.z % uniforms.num_heads;
   if (m < uniforms.M && n < uniforms.N) {
     let outputIdx = batchIdx * uniforms.M * uniforms.v_hidden_size + m * uniforms.v_hidden_size
       + currentBatchHeadNumber * uniforms.N + n;
     output[outputIdx] = value;
   }
  }`;
  };

  return {
    name: 'AttentionScore',
    shaderCache: { hint: `${pastValue !== undefined};${outputCount}`, inputDependencies },
    getRunData: () => ({ outputs, dispatchGroup: dispatch, programUniforms }),
    getShaderSource,
  };
};

export const applyAttention = (
  context: ComputeContext,
  q: TensorView,
  k: TensorView,
  v: TensorView,
  _maskIndex: TensorView | undefined,
  _past: TensorView | undefined,
  pastKey: TensorView | undefined,
  pastValue: TensorView | undefined,
  attentionBiasInput: TensorView | undefined,
  parameters: AttentionParameters,
  attributes: AttentionAttrs,
) => {
  // Assumption  is that presentKey/presentValue exists only if pastKey/pastValue exists.
  const outputCount = Math.min(context.outputCount, 1 + (pastKey ? 1 : 0) + (pastValue ? 1 : 0));
  const pastSequenceLength = parameters.kvNumHeads !== undefined || outputCount > 1 ? parameters.pastSequenceLength : 0;
  const totalSequenceLength = pastSequenceLength + parameters.kvSequenceLength;
  const attentionBias =
    attentionBiasInput && ShapeUtil.size(attentionBiasInput.dims) > 0 ? attentionBiasInput : undefined;

  const inputsK = [q, k];
  if (parameters.kvNumHeads === undefined && outputCount > 1 && pastKey && ShapeUtil.size(pastKey.dims) > 0) {
    inputsK.push(pastKey);
  }
  if (attentionBias) {
    inputsK.push(attentionBias);
  }

  // Run AttentionProbs
  const probs = context.compute(
    createAttentionProbsProgramInfo(
      outputCount,
      q,
      k,
      pastKey,
      attentionBias,
      parameters,
      attributes,
      pastSequenceLength,
    ),
    { inputs: inputsK, outputs: parameters.kvNumHeads === undefined && outputCount > 1 ? [-1, 1] : [-1] },
  )[0];

  // Run Softmax
  context.compute(
    createInPlaceSoftmaxProgramInfo(
      probs,
      parameters.batchSize * parameters.numHeads * parameters.sequenceLength,
      totalSequenceLength,
    ),
    { inputs: [probs], outputs: [] },
  );

  // Run AttrionScore
  const inputsV = [probs, v];
  if (parameters.kvNumHeads === undefined && outputCount > 1 && pastValue && ShapeUtil.size(pastValue.dims) > 0) {
    inputsV.push(pastValue);
  }
  context.compute(createVxAttentionScoreProgramInfo(outputCount, probs, v, pastValue, parameters, pastSequenceLength), {
    inputs: inputsV,
    outputs: parameters.kvNumHeads === undefined && outputCount > 1 ? [0, 2] : [0],
  });
};

const prepare = (context: ComputeContext, parameters: AttentionParameters) => {
  const outputShape = [parameters.batchSize, parameters.numHeads, parameters.sequenceLength, parameters.headSize];
  const M = parameters.sequenceLength;
  const K = parameters.inputHiddenSize;
  const N = parameters.headSize;
  const TILE_SIZE = 12;
  const dispatch = {
    x: Math.ceil(parameters.headSize / TILE_SIZE),
    y: Math.ceil(parameters.sequenceLength / TILE_SIZE),
    z: parameters.batchSize * parameters.numHeads,
  };
  const inputs = [context.inputs[0], context.inputs[1], context.inputs[2]];
  const programUniforms: ProgramUniform[] = [
    { type: DataType.uint32, data: M },
    { type: DataType.uint32, data: K },
    { type: DataType.uint32, data: N },
    { type: DataType.uint32, data: parameters.numHeads },
    { type: DataType.uint32, data: parameters.headSize },
    { type: DataType.uint32, data: parameters.hiddenSize },
    { type: DataType.uint32, data: parameters.hiddenSize + parameters.hiddenSize + parameters.vHiddenSize },
  ];

  const getShaderSource = (shaderHelper: ShaderHelper) => {
    const outputQ = outputVariable('output_q', inputs[0].dataType, outputShape);
    const outputK = outputVariable('output_k', inputs[0].dataType, outputShape);
    const outputV = outputVariable('output_v', inputs[0].dataType, outputShape);
    const input = inputVariable('input', inputs[0].dataType, inputs[0].dims);
    const weight = inputVariable('weight', inputs[1].dataType, inputs[1].dims);
    const bias = inputVariable('bias', inputs[2].dataType, inputs[2].dims);
    const dataType = input.type.storage;

    const uniforms: UniformsArrayType = [
      { name: 'M', type: 'u32' },
      { name: 'K', type: 'u32' },
      { name: 'N', type: 'u32' },
      { name: 'num_heads', type: 'u32' },
      { name: 'head_size', type: 'u32' },
      { name: 'hidden_size', type: 'u32' },
      { name: 'ldb', type: 'u32' },
    ];
    return `
  const TILE_SIZE = ${TILE_SIZE}u;
  var<workgroup> tileInput: array<${dataType}, ${TILE_SIZE * TILE_SIZE}>;
  var<workgroup> tileWeightQ: array<${dataType}, ${TILE_SIZE * TILE_SIZE}>;
  var<workgroup> tileWeightK: array<${dataType}, ${TILE_SIZE * TILE_SIZE}>;
  var<workgroup> tileWeightV: array<${dataType}, ${TILE_SIZE * TILE_SIZE}>;
  ${shaderHelper.registerUniforms(uniforms).declareVariables(input, weight, bias, outputQ, outputK, outputV)}
  ${shaderHelper.mainStart([TILE_SIZE, TILE_SIZE, 1])}
    let batchIndex = workgroup_id.z / uniforms.num_heads;
    let headNumber = workgroup_id.z % uniforms.num_heads;
    let m = global_id.y;
    let n = global_id.x;

    let inputOffset = batchIndex * (uniforms.M * uniforms.K) + m * uniforms.K;
    let biasOffsetQ = headNumber * uniforms.head_size;
    let biasOffsetK = uniforms.hidden_size + biasOffsetQ;
    let biasOffsetV = uniforms.hidden_size + biasOffsetK;

    var valueQ = ${dataType}(0);
    var valueK = ${dataType}(0);
    var valueV = ${dataType}(0);
    for (var w: u32 = 0u; w < uniforms.K; w += TILE_SIZE) {
      if (m < uniforms.M && w + local_id.x < uniforms.K) {
        tileInput[TILE_SIZE * local_id.y + local_id.x] = input[inputOffset + w + local_id.x];
      }
      if (n < uniforms.N && w + local_id.y < uniforms.K) {
        let offset = n + (w + local_id.y) * uniforms.ldb;
        tileWeightQ[TILE_SIZE * local_id.y + local_id.x] = weight[biasOffsetQ + offset];
        tileWeightK[TILE_SIZE * local_id.y + local_id.x] = weight[biasOffsetK + offset];
        tileWeightV[TILE_SIZE * local_id.y + local_id.x] = weight[biasOffsetV + offset];
      }
      workgroupBarrier();
      for (var k: u32 = 0u; k<TILE_SIZE && w+k < uniforms.K; k++) {
        let inputTileOffset = TILE_SIZE * local_id.y + k;
        let weightTileOffset = TILE_SIZE * k + local_id.x;
        valueQ += tileInput[inputTileOffset] * tileWeightQ[weightTileOffset];
        valueK += tileInput[inputTileOffset] * tileWeightK[weightTileOffset];
        valueV += tileInput[inputTileOffset] * tileWeightV[weightTileOffset];
      }

      workgroupBarrier();
    }

    let headOffset = (m * uniforms.N + n) % uniforms.head_size;
    valueQ += bias[headOffset + biasOffsetQ];
    valueK += bias[headOffset + biasOffsetK];
    valueV += bias[headOffset + biasOffsetV];

    let offset = workgroup_id.z * uniforms.M * uniforms.N;
    if (m < uniforms.M && n < uniforms.N) {
      let outputIdx = offset + m * uniforms.N + n;
      output_q[outputIdx] = valueQ;
      output_k[outputIdx] = valueK;
      output_v[outputIdx] = valueV;
    }
  }`;
  };

  return context.compute(
    {
      name: 'AttentionPrepare',
      shaderCache: { inputDependencies: ['type', 'type', 'type'] },
      getRunData: () => ({
        outputs: [
          { dims: outputShape, dataType: context.inputs[0].dataType, gpuDataType: GpuDataType.default },
          { dims: outputShape, dataType: context.inputs[0].dataType, gpuDataType: GpuDataType.default },
          { dims: outputShape, dataType: context.inputs[0].dataType, gpuDataType: GpuDataType.default },
        ],
        dispatchGroup: dispatch,
        programUniforms,
      }),
      getShaderSource,
    },
    { inputs, outputs: [-1, -1, -1] },
  );
};

export const attention = (context: ComputeContext, attributes: AttentionAttrs): void => {
  const params = validateAttentionInputs(context.inputs, attributes);

  const [q, k, v] = prepare(context, params);

  return applyAttention(
    context,
    q,
    k,
    v,
    context.inputs[4],
    undefined,
    undefined,
    undefined,
    context.inputs[5],
    params,
    attributes,
  );
};
