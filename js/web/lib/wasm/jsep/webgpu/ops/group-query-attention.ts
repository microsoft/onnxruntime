// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import { DataType } from '../../../wasm-common';
import { TensorView } from '../../tensor-view';
import { ShapeUtil } from '../../util';
import { createAttributeWithCacheKey } from '../attribute-with-cache-key';
import { ComputeContext, ProgramInfo, ProgramInputTensorInfoDependency, ProgramUniform } from '../types';

import {
  applyAttention,
  AttentionAttrs,
  AttentionMaskType,
  AttentionParameters,
  AttentionQkvFormat,
} from './attention';
import { createTensorShapeVariables, inputVariable, outputVariable, ShaderHelper, UniformsArrayType } from './common';
import { maybeTransposeToBNSHAndAddBias } from './multihead-attention';
import { createTileProgramInfo } from './tile';
import { createTransposeProgramInfo, TransposeAttributes } from './transpose';

export const validateInputs = (inputs: readonly TensorView[], attributes: AttentionAttrs): AttentionParameters => {
  const query = inputs[0];
  const key = inputs[1];
  const value = inputs[2];
  const pastKey = inputs[3];
  const pastValue = inputs[4];

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

  //     past_key                   : (B, N, S*, H)
  //     past_value                 : (B, N, S*, H)
  // When no packing for q/k/v:
  //     query            (Q)       : (B, S, D)
  //     key              (K)       : (B, L, D) or (B, N, S*, H)
  //     value            (V)       : (B, L, D_v) or (B, N, S*, H)
  // When packed kv is used:
  //     query            (Q)       : (B, S, D)
  //     key              (K)       : (B, L, N, 2, H)
  //     value            (V)       : None
  // When packed qkv is used:
  //     query            (Q)       : (B, L, N, 3, H) or (B, S, 3*D)
  //     key              (K)       : None
  //     value            (V)       : None

  if (query.dims.length !== 3 && query.dims.length !== 5) {
    throw new Error('Input query is expected to have 3 or 5 dimensions');
  }

  const dmmhaPacking = false;
  const batchSize = query.dims[0];
  const sequenceLength = query.dims[1];
  const hiddenSize =
    query.dims.length === 3 ? (dmmhaPacking ? query.dims[2] / 3 : query.dims[2]) : attributes.numHeads * query.dims[4];
  let kvSequenceLength = sequenceLength;

  let pastSequenceLength = 0;
  let maxSequenceLength = 0;
  const headSize = Math.floor(hiddenSize / attributes.numHeads);
  const hasPastKey = pastKey && pastKey.dims.length !== 0;
  const hasPastValue = pastValue && pastValue.dims.length !== 0;
  // TODO : this should be from attributes.
  const isPastkvBSNH = true;
  if (hasPastKey && hasPastValue) {
    if (pastKey.dims.length !== 4) {
      throw new Error('Input "past_key" is expected to have 4 dimensions');
    }
    if (pastValue.dims.length !== 4) {
      throw new Error('Input "past_value" is expected to have 4 dimensions');
    }
    if (isPastkvBSNH) {
      // For BSNH
      pastSequenceLength = pastKey.dims[1];
      maxSequenceLength = pastKey.dims[1];
    } else {
      // For BNSH
      pastSequenceLength = pastKey.dims[2];
      maxSequenceLength = pastKey.dims[2];
    }
  } else if (hasPastKey || hasPastValue) {
    throw new Error('Input "past_key" and "past_value" shall be both present or both absent');
  }

  let qkvFormat: AttentionQkvFormat;
  if (key) {
    if (query.dims.length !== 3) {
      throw new Error('Input "query" is expected to have 3 dimensions when key is given');
    }
    if (key.dims.length < 3 || key.dims.length > 5) {
      throw new Error('Input "key" is expected to have 3, 4, or 5 dimensions');
    }
    if (query.dims[0] !== key.dims[0]) {
      throw new Error('Input "query" and "key" shall have same dim 0 (batch size)');
    }

    if (key.dims.length === 3) {
      if (query.dims[2] % key.dims[2] !== 0) {
        throw new Error('Dimension 2 of "query" should be a multiple of "key"');
      }
      qkvFormat = AttentionQkvFormat.qkvBSNH;
      kvSequenceLength = key.dims[1];
    } else if (key.dims.length === 5) {
      if (key.dims[2] !== attributes.numHeads || key.dims[3] !== 2 || key.dims[4] !== headSize) {
        throw new Error('Expect "key" shape (batch_size, kv_sequence_length, num_heads, 2, head_size) for packed kv');
      }
      if (value) {
        throw new Error('Expect "value" be none when "key" has packed kv format.');
      }
      qkvFormat = AttentionQkvFormat.qKvBSNHxBSN2H;
      kvSequenceLength = key.dims[1];
    } else {
      // key_dims.size() == 4 (cross-attention with past_key)
      if (key.dims[1] !== attributes.numHeads || key.dims[3] !== headSize) {
        throw new Error('Expect "key" shape (batch_size, num_heads, kv_sequence_length, head_size) for past_key');
      }

      qkvFormat = AttentionQkvFormat.unknown;
      kvSequenceLength = key.dims[2];
    }
  } else {
    // packed QKV
    if (query.dims.length !== 3 && query.dims.length !== 5) {
      throw new Error('Input "query" is expected to have 3 or 5 dimensions when key is empty');
    }
    if (query.dims.length === 5 && (query.dims[2] !== attributes.numHeads || query.dims[3] !== 3)) {
      throw new Error('Expect "query" shape (batch_size, kv_sequence_length, num_heads, 3, head_size) for packed kv');
    }

    qkvFormat = AttentionQkvFormat.qkvBSN3H;
  }

  const maskType: AttentionMaskType = AttentionMaskType.none;
  let passPastInKv = false;
  let vHiddenSize = hiddenSize;
  if (value) {
    if (value.dims.length !== 3 && value.dims.length !== 4) {
      throw new Error('Input "value" is expected to have 3 or 4 dimensions');
    }

    if (query.dims[0] !== value.dims[0]) {
      throw new Error('Input "query" and "value" shall have same dim 0 (batch_size)');
    }

    if (value.dims.length === 3) {
      if (kvSequenceLength !== value.dims[1]) {
        throw new Error('Input "key" and "value" shall have the same dim 1 (kv_sequence_length)');
      }
      vHiddenSize = value.dims[2];
    } else {
      if (kvSequenceLength !== value.dims[2]) {
        throw new Error('Input "past_key" and "past_value" shall have the same dim 2 (kv_sequence_length)');
      }
      vHiddenSize = value.dims[1] * value.dims[3];
      passPastInKv = true;
    }
  }
  const totalSequenceLength = pastSequenceLength + kvSequenceLength;
  const broadcastResPosBias = false;

  return {
    batchSize,
    sequenceLength,
    pastSequenceLength,
    kvSequenceLength,
    totalSequenceLength,
    maxSequenceLength,
    inputHiddenSize: 0,
    hiddenSize,
    vHiddenSize,
    headSize,
    vHeadSize: Math.floor(vHiddenSize / attributes.kvNumHeads!),
    numHeads: attributes.numHeads,
    kvNumHeads: attributes.kvNumHeads,
    nReps: attributes.numHeads / attributes.kvNumHeads!,
    pastPresentShareBuffer: false,
    maskType,
    scale: attributes.scale,
    broadcastResPosBias,
    passPastInKv,
    qkvFormat,
    isPastkvBSNH,
  };
};

const createConcatProgramInfo = (
  a: TensorView,
  b: TensorView | undefined,
  dataType: DataType,
  params: AttentionParameters,
): ProgramInfo => {
  const outputShape = [params.batchSize, params.totalSequenceLength, params.kvNumHeads!, params.headSize];
  const component = 4;
  const outputSize = ShapeUtil.size(outputShape) / component;
  const presentSequenceLength = params.totalSequenceLength;
  const output = outputVariable('present_kv', dataType, outputShape.length, component);
  const inputA = inputVariable('new_kv', a.dataType, a.dims.length, component);
  const inputB = b ? inputVariable('past_kv', b.dataType, b.dims.length, component) : undefined;

  const H = Math.ceil(params.headSize / component);
  const dispatch = { x: presentSequenceLength, y: a.dims[0], z: 1 };

  const inputDependencies: ProgramInputTensorInfoDependency[] = b ? ['rank', 'rank'] : ['rank'];

  const programUniforms: ProgramUniform[] = [
    { type: DataType.uint32, data: outputSize },
    { type: DataType.uint32, data: params.pastSequenceLength },
    { type: DataType.uint32, data: params.kvSequenceLength },
    { type: DataType.uint32, data: params.totalSequenceLength },
  ];

  const inputs = [inputA];
  if (inputB) {
    programUniforms.push(
      ...createTensorShapeVariables(a.dims),
      ...createTensorShapeVariables(b!.dims),
      ...createTensorShapeVariables(outputShape),
    );
    inputs.push(inputB);
  } else {
    programUniforms.push(...createTensorShapeVariables(a.dims), ...createTensorShapeVariables(outputShape));
  }
  const uniforms: UniformsArrayType = [
    { name: 'output_size', type: 'u32' },
    { name: 'past_seqlen', type: 'u32' },
    { name: 'new_seqlen', type: 'u32' },
    { name: 'present_seqlen', type: 'u32' },
  ];

  const pastStr = `      let past_batch_stride = uniforms.past_seqlen * num_heads * H;
        var past_head_stride = uniforms.past_seqlen * H;
        if (is_bsnh) {
          past_head_stride = H;
        }
        let in_offset = b * past_batch_stride + s * row_stride + n * past_head_stride + h;
        present_kv[out_offset] = past_kv[in_offset];`;
  const newStr = `      let new_batch_stride = uniforms.new_seqlen * num_heads * H;
        let new_row_stride = num_heads * H;
        let new_head_stride = H;
        let in_offset = b * new_batch_stride + (s - past_seqlen) * new_row_stride + n * new_head_stride + h;
        present_kv[out_offset] = new_kv[in_offset];`;
  const concatStr = b
    ? `if (s < past_seqlen) {
        ${pastStr}
        } else if (s < past_seqlen + uniforms.new_seqlen) {
        ${newStr}
        }`
    : `if (s < past_seqlen + uniforms.new_seqlen) {
          ${newStr}
        }`;

  // TODO: handle H * params.kvNumHeads greater than maxComputeInvocationsPerWorkgroup limit.
  const getShaderSource = (shaderHelper: ShaderHelper) => `

  ${shaderHelper.registerUniforms(uniforms).declareVariables(...inputs, output)}
  ${shaderHelper.mainStart([H, params.kvNumHeads!, 1])}
    ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes('uniforms.output_size')}
    var indices = ${output.offsetToIndices('global_idx')};
    let h = local_id.x;
    let n = local_id.y;
    let s = workgroup_id.x;
    let b = workgroup_id.y;
    let num_heads = ${params.kvNumHeads!}u;
    let H = ${H}u;

    let present_seqlen = uniforms.present_seqlen;
    let present_batch_stride = present_seqlen * num_heads * H;
    var row_stride = H;
    let is_bsnh = ${params.isPastkvBSNH};

    if (is_bsnh) {
      row_stride = num_heads * H;
    }
    var present_head_stride = present_seqlen * H;
    if (is_bsnh) {
      present_head_stride = H;
    }

    let past_seqlen = uniforms.past_seqlen;

    let out_offset = b * present_batch_stride + s * row_stride + n * present_head_stride + h;
    ${concatStr}
  }`;

  return {
    name: 'ConcatPastNew',
    shaderCache: { hint: `${params.kvNumHeads!}${H}${!!b}`, inputDependencies },
    getRunData: () => ({
      outputs: [{ dims: outputShape, dataType }],
      dispatchGroup: dispatch,
      programUniforms,
    }),
    getShaderSource,
  };
};

export const parseGroupQueryAttentionAttributes = (attributes: AttentionAttrs): AttentionAttrs =>
  createAttributeWithCacheKey({ ...attributes });

const weightTransposeAttribute: TransposeAttributes = createAttributeWithCacheKey({ perm: [0, 2, 1, 3] });

const maybeExpandAndTransposeToBNSH = (
  context: ComputeContext,
  input: TensorView,
  pastKV: TensorView | undefined,
  params: AttentionParameters,
  outputIndex: number,
) => {
  let reshapedInput = input;
  const numHeads = params.kvNumHeads!;
  const nReps = params.nReps!;
  if (input.dims.length === 3 && params.kvSequenceLength !== 0) {
    reshapedInput = input.reshape([params.batchSize, params.kvSequenceLength, numHeads, params.headSize]);
  }

  if (pastKV) {
    reshapedInput = context.compute(createConcatProgramInfo(reshapedInput, pastKV, reshapedInput.dataType, params), {
      inputs: [reshapedInput, pastKV],
      outputs: [params.isPastkvBSNH ? outputIndex : -1],
    })[0];
  } else {
    reshapedInput = context.compute(createConcatProgramInfo(reshapedInput, undefined, reshapedInput.dataType, params), {
      inputs: [reshapedInput],
      outputs: [params.isPastkvBSNH ? outputIndex : -1],
    })[0];
  }
  if (nReps !== 1) {
    reshapedInput = context.compute(createTileProgramInfo([reshapedInput], [1, 1, 1, nReps]), {
      inputs: [reshapedInput],
      outputs: [-1],
    })[0];
    reshapedInput = reshapedInput.reshape([
      params.batchSize,
      params.totalSequenceLength,
      numHeads * nReps,
      params.headSize,
    ]);
  }

  return context.compute(createTransposeProgramInfo(reshapedInput, weightTransposeAttribute.perm), {
    inputs: [reshapedInput],
    outputs: [-1],
  })[0];
};

export const groupQueryAttention = (context: ComputeContext, attributes: AttentionAttrs): void => {
  const params = validateInputs(context.inputs, attributes);
  if (context.inputs[0].dims.length === 5) {
    throw new Error('Packed QKV is not implemented');
  }

  if (context.inputs[1]?.dims.length === 5) {
    throw new Error('Packed KV is not implemented');
  }

  const Q = maybeTransposeToBNSHAndAddBias(
    context,
    params.batchSize,
    params.numHeads,
    params.sequenceLength,
    params.headSize,
    context.inputs[0],
    undefined,
    0,
  );
  const pastKey = context.inputs[3] && context.inputs[3].dims.length !== 0 ? context.inputs[3] : undefined;
  const pastValue = context.inputs[4] && context.inputs[4].dims.length !== 0 ? context.inputs[4] : undefined;
  const K = maybeExpandAndTransposeToBNSH(context, context.inputs[1], pastKey, params, 1);
  const V = maybeExpandAndTransposeToBNSH(context, context.inputs[2], pastValue, params, 2);
  applyAttention(context, Q, K, V, undefined, undefined, undefined, undefined, undefined, params, attributes);
};
