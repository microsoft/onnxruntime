// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {DataType} from '../../../wasm-common';
import {TensorView} from '../../tensor';
import {GemmUtil, ShapeUtil} from '../../util';
import {AttributeWithCacheKey, createAttributeWithCacheKey} from '../attribute-with-cache-key';
import {ComputeContext, GpuDataType, ProgramInfo, ProgramInfoLoader, ProgramMetadata} from '../types';

import {ShaderHelper} from './common';

const validateInputs = (inputs: readonly TensorView[]): void => {
  if (!inputs) {
    throw new Error('Input is missing');
  }
  if (inputs.length < 2 || inputs.length > 3) {
    throw new Error('Invaid input number.');
  }

  // 'C' can be of dimensionality 0, 1 or 2 only
  if (inputs.length === 3 && inputs[2].dims.length > 2) {
    throw new Error('Invalid input shape of C');
  }

  if ((inputs[0].dataType !== DataType.float) || (inputs[1].dataType !== DataType.float) ||
      (inputs.length === 3 && inputs[2].dataType !== DataType.float)) {
    throw new Error('Invalid input type.');
  }

  if ((inputs[0].dataType !== inputs[1].dataType) ||
      (inputs.length === 3 && inputs[0].dataType !== inputs[2].dataType)) {
    throw new Error('Input types are mismatched');
  }
};

export interface GemmAttributes extends AttributeWithCacheKey {
  transA: boolean;
  transB: boolean;
  alpha: number;
  beta: number;
}

const offsetC = (m: number, n: number, dims: readonly number[]): string => {
  if (dims.length === 0) {
    return '0u';
  }

  const broadcastM = (dims.length === 1 && m !== 1) || (dims.length === 2 && dims[0] !== m);
  const broadcastN = dims[dims.length - 1] !== n;

  let offset = '0u';
  if (!broadcastM) {
    offset += `+ m * ${dims[dims.length - 1]}u`;
  }
  if (!broadcastN) {
    offset += '+n';
  }

  return offset;
};

const createGemmProgramInfo =
    (metadata: ProgramMetadata, inputs: readonly TensorView[], attributes: GemmAttributes): ProgramInfo => {
      const aShape = inputs[0].dims.slice();
      const bShape = inputs[1].dims.slice();
      const [M, N, K] = GemmUtil.getShapeOfGemmResult(
          aShape, attributes.transA, bShape, attributes.transB, inputs.length === 3 ? inputs[2].dims : undefined);
      const outputShape = [M, N];
      if (!outputShape) {
        throw new Error('Can\'t use gemm on the given tensors');
      }
      const outputSize = ShapeUtil.size(outputShape);
      let line = '';
      if (attributes.transA && attributes.transB) {
        line = 'value += a[k * M + m] * b[n * K + k];';
      } else if (attributes.transA && !attributes.transB) {
        line = 'value += a[k * M + m] * b[k * N + n];';
      } else if (!attributes.transA && attributes.transB) {
        line = 'value += a[m * K + k] * b[n * K + k];';
      } else if (!attributes.transA && !attributes.transB) {
        line = 'value += a[m * K + k] * b[k * N + n];';
      }

      const dataType = 'f32';  // TODO: support other data type
      const calculateAlpha = attributes.alpha === 1 ? '' : 'value *= alpha;';
      const calculateC = inputs.length === 3 ? `value += beta * c[${offsetC(M, N, inputs[2].dims)}];` : '';
      const inputStorageBuffersDeclarations = [
        `@group(0) @binding(0) var<storage, read> a : array<${dataType}>;`,
        `@group(0) @binding(1) var<storage, read> b : array<${dataType}>;`
      ];
      if (inputs.length === 3) {
        inputStorageBuffersDeclarations.push(`@group(0) @binding(2) var<storage, read> c : array<${dataType}>;`);
      }
      const getShaderSource = (shaderHelper: ShaderHelper) => `
  const M: u32 = ${M}u;
  const N: u32 = ${N}u;
  const K: u32 = ${K}u;
  const alpha = ${dataType}(${attributes.alpha});
  const beta = ${dataType}(${attributes.beta});

  ${inputStorageBuffersDeclarations.join('\n')}
  @group(0) @binding(${inputs.length}) var<storage, read_write> output : array<${dataType}>;

  ${shaderHelper.mainStart()}
    ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes(outputSize)}

    let m = global_id.x / N;
    let n = global_id.x % N;

    var value = ${dataType}(0);
    for (var k: u32 = 0u; k<${K}u; k++) {
      ${line}
    }

    ${calculateAlpha}
    ${calculateC}
    output[global_id.x] = value;

  }`;
      return {
        ...metadata,
        outputs: [{dims: outputShape, dataType: inputs[0].dataType, gpuDataType: GpuDataType.default}],
        getShaderSource,
        dispatchGroup: () => ({x: Math.ceil(outputSize / 64 /* workgroup size */)})
      };
    };

const createGemmProgramInfoLoader = (inputs: readonly TensorView[], attributes: GemmAttributes): ProgramInfoLoader => {
  const metadata = {
    name: 'Gemm',
    inputTypes: inputs.length === 3 ? [GpuDataType.default, GpuDataType.default, GpuDataType.default] :
                                      [GpuDataType.default, GpuDataType.default],
    cacheHint: attributes.cacheKey
  };

  return {...metadata, get: () => createGemmProgramInfo(metadata, inputs, attributes)};
};

export const gemm = (context: ComputeContext, attributes: GemmAttributes): void => {
  validateInputs(context.inputs);
  context.compute(createGemmProgramInfoLoader(context.inputs, attributes));
};

export const parseGemmAttributes = (attributes: Record<string, unknown>): GemmAttributes =>
    createAttributeWithCacheKey(attributes as Omit<GemmAttributes, keyof AttributeWithCacheKey>);
