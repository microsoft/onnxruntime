// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {DataType} from '../../../wasm-common';
import {TensorView} from '../../tensor';
import {ShapeUtil} from '../../util';
import {createAttributeWithCacheKey} from '../attribute-with-cache-key';
import {ComputeContext, GpuDataType, ProgramInfo, ProgramInfoLoader, ProgramMetadata} from '../types';

import {createIndicesHelper, ShaderHelper} from './common';
import {ConvAttributes} from './conv';
import {parseInternalActivationAttributes} from './fuse-utils';

export interface ConvTransposeAttributes extends ConvAttributes {
  readonly outputPadding: readonly number[];
  readonly outputShape: readonly number[];
}

export const parseConvTransposeAttributes = (attributes: Record<string, unknown>): ConvTransposeAttributes => {
  const activationAttributes = parseInternalActivationAttributes(attributes);
  // TODO : Make this generic enough to compute default attributes for multi-dimensional conv
  const format = attributes.format as 'NHWC' | 'NCHW';
  const autoPad = ['NOTSET', 'VALID', 'SAME_UPPER', 'SAME_LOWER'][attributes.auto_pad as number];
  const dilations = attributes.dilations as [number, number];
  const group = attributes.group as number;
  const kernelShape = attributes.kernel_shape as [number, number];
  const pads = attributes.pads as [number, number, number, number];
  const strides = attributes.strides as [number, number];
  const wIsConst = (attributes.w_is_const as () => boolean)();
  const outputPadding = attributes.output_padding as [number, number, number, number];
  const outputShape = attributes.output_shape as [number, number];
  return createAttributeWithCacheKey({
    autoPad,
    format,
    dilations,
    group,
    kernelShape,
    outputPadding,
    outputShape,
    pads,
    strides,
    wIsConst,
    ...activationAttributes
  });
};

const validateInputs = (inputs: readonly TensorView[], attributes: ConvTransposeAttributes): void => {
  // Refer to the below link for all input checks
  // https://github.com/onnx/onnx/blob/main/docs/Operators.md#ConvTranspose
  if (!inputs || (inputs.length !== 2 && inputs.length !== 3)) {
    throw new Error('Conv requires 2 or 3 inputs');
  }

  // TODO : Need to add support for multi-dimensional conv
  if (inputs[0].dims.length !== 4 || inputs[1].dims.length !== 4) {
    throw new Error('currently only support 2-dimensional conv');
  }

  // FILTER_IN_CHANNEL should be equal to DATA_CHANNEL
  const dataChannel = inputs[0].dims[1];
  const filterInChannel = inputs[1].dims[0];
  if (dataChannel !== filterInChannel) {
    throw new Error('FILTER_IN_CHANNEL should be equal to DATA_CHANNEL');
  }

  const featureMaps = inputs[1].dims[1] * attributes.group;

  // if bias is provided it should be 1D and the number of elements should be equal to the number of feature maps
  if (inputs.length === 3 && (inputs[2].dims.length !== 1 || inputs[2].dims[0] !== featureMaps)) {
    throw new Error('invalid bias');
  }

  const spatialRank = inputs[0].dims.length - 2;
  // wrong dilations dimension
  if (attributes.dilations.length !== spatialRank) {
    throw new Error(`dilations should be ${spatialRank}D`);
  }

  // Wrong strides dimension
  if (attributes.strides.length !== spatialRank) {
    throw new Error(`strides should be ${spatialRank}D`);
  }

  // Wrong pads dimension
  if (attributes.pads.length !== spatialRank * 2) {
    throw new Error(`pads should be ${spatialRank * 2}D`);
  }

  // Wrong output padding dimension
  if (attributes.outputPadding.length !== spatialRank) {
    throw new Error(`output_padding should be ${spatialRank}D`);
  }

  // if kernelShape is specified, it's data length must be 2 less than dims length of the weights tensor
  // (the first 2 dims are batch_size and channels)
  if (attributes.kernelShape.length !== 0 && attributes.kernelShape.length !== inputs[1].dims.length - 2) {
    throw new Error('invalid kernel shape');
  }

  // as with kernelShape, must have same number of spatial dims as input
  if (attributes.outputShape.length !== 0 && attributes.outputShape.length !== inputs[0].dims.length - 2) {
    throw new Error('invalid output shape');
  }

  // TODO : Need to add support for float64
  if (inputs[0].dataType !== DataType.float || inputs[1].dataType !== DataType.float) {
    throw new Error('ConvTranspose input(X,W) should be float tensor');
  }

  if (inputs.length === 3 && inputs[2].dataType !== DataType.float) {
    throw new Error('ConvTranspose input(bias) should be float tensor');
  }
};

const createConvTransposeProgramMetadata = (hasBias: boolean, cacheHint: string): ProgramMetadata => ({
  name: 'ConvTranspose',
  inputTypes: hasBias ? [GpuDataType.default, GpuDataType.default, GpuDataType.default] :
                        [GpuDataType.default, GpuDataType.default],
  cacheHint
});

const createConvTransposeProgramInfo =
    (inputs: readonly TensorView[], metadata: ProgramMetadata, attributes: ConvTransposeAttributes): ProgramInfo => {
      const hasBias = inputs.length > 2;
      const processBias = hasBias ? 'value += b[output_channel];' : '';
      const xShape = inputs[0].dims;
      const wShape = inputs[1].dims;
      const dataType = 'f32';  // TODO: support other data type
      const outputShape = attributes.outputShape;
      const outputSize = ShapeUtil.size(outputShape);
      const outputIndicesHelper = createIndicesHelper('output', outputShape);
      const xIndicesHelper = createIndicesHelper('x', xShape);
      const wIndicesHelper = createIndicesHelper('w', wShape);
      const inputStorageBuffersDeclarations = [
        `@group(0) @binding(0) var<storage, read> x : array<${dataType}>;`,
        `@group(0) @binding(1) var<storage, read> w : array<${dataType}>;`
      ];
      if (hasBias) {
        inputStorageBuffersDeclarations.push(`@group(0) @binding(2) var<storage, read> b : array<${dataType}>;`);
      }
      const getShaderSource = (shaderHelper: ShaderHelper) => `
      ${inputStorageBuffersDeclarations.join('\n')}
      @group(0) @binding(${inputStorageBuffersDeclarations.length}) var<storage, read_write> output : array<${
          dataType}>;
      ${outputIndicesHelper.o2iImpl}
      ${xIndicesHelper.i2oImpl}
      ${wIndicesHelper.i2oImpl}

      ${shaderHelper.mainStart()}
        ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes(outputSize)}
        var value: ${dataType} = ${dataType}(0);

        ${outputIndicesHelper.indicesVariableDeclaration('outputIndices')}
        ${outputIndicesHelper.o2iCall('global_idx', 'outputIndices')}
        ${processBias}
        output[global_idx] = value;
      }`;
      return {
        ...metadata,
        outputs: [],
        getShaderSource,
        dispatchGroup: () => ({x: Math.ceil(outputSize / 64 /* workgroup size */)})
      };
    };

export const createConvTransposeProgramInfoLoader =
    (inputs: readonly TensorView[], attributes: ConvTransposeAttributes): ProgramInfoLoader => {
      const metadata = createConvTransposeProgramMetadata(inputs.length > 2, attributes.cacheKey);
      return {...metadata, get: () => createConvTransposeProgramInfo(inputs, metadata, attributes)};
    };
const convTranspose2d =
    (context: ComputeContext, inputs: readonly TensorView[], attributes: ConvTransposeAttributes): void => {

    };

export const convTranspose = (context: ComputeContext, attributes: ConvTransposeAttributes): void => {
  validateInputs(context.inputs, attributes);
  convTranspose2d(context, context.inputs, attributes);
}
