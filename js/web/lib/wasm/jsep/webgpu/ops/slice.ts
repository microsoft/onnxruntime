// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {AttributeWithCacheKey, createAttributeWithCacheKey} from '../../../attribute-with-cache-key';
import {Graph} from '../../../graph';
import {NUMBER_TYPES, OperatorAsyncImplementation, OperatorInitialization} from '../../../operators';
import {Tensor} from '../../../tensor';
import {ShapeUtil} from '../../../util';
import {WebGpuInferenceHandler} from '../inference-handler';
import {GpuDataType, ProgramInfo} from '../types';

import {WORKGROUP_SIZE} from './common';

export interface SliceAttributes extends AttributeWithCacheKey {
  readonly axes: number[];
  readonly ends: number[];
  readonly starts: number[];
}

const sliceProgramMetadata = {
  name: 'Slice',
  inputTypes: [GpuDataType.default]
};

export const slice: OperatorAsyncImplementation<SliceAttributes> = async(
    inferenceHandler: WebGpuInferenceHandler, inputs: Tensor[], attributes: SliceAttributes): Promise<Tensor[]> => {
  validateInputs(inputs);
  return inferenceHandler.run(
      {
        ...sliceProgramMetadata,
        cacheHint: attributes.cacheKey,
        get: () => createSliceProgramInfo(inputs[0], attributes)
      },
      inputs);
};

export const parseSliceAttributes: OperatorInitialization<SliceAttributes> = (node: Graph.Node): SliceAttributes => {
  const starts = node.attributes.getInts('starts');
  const ends = node.attributes.getInts('ends');
  const axes = node.attributes.getInts('axes', []);
  return createAttributeWithCacheKey({starts, ends, axes});
};

const offsetToIndices = (offset: string, strides: readonly number[], indicesPrefix: string): string => {
  const outputLines: string[] = [];

  for (let i = 0; i < strides.length - 1; i++) {
    outputLines.push(`var ${indicesPrefix}${i}=${offset}/${strides[i]}u;`);
    outputLines.push(`${offset}%=${strides[i]}u;`);
  }
  outputLines.push(`var ${indicesPrefix}${strides.length - 1}=${offset};`);

  return outputLines.join('\n');
};

const indicesToOffset = (indicesPrefix: string, strides: readonly number[], offset: string): string => {
  const outputLines: string[] = [];

  for (let i = 0; i < strides.length - 1; i++) {
    outputLines.push(`${offset}+=${indicesPrefix}${i} * ${strides[i]}u;`);
  }
  outputLines.push(`${offset}+=${indicesPrefix}${strides.length - 1};`);

  return outputLines.join('\n');
};

const createSliceProgramInfo = (input: Tensor, attributes: SliceAttributes, dataType = 'f32'): ProgramInfo => {
  const axes = (attributes.axes.length === 0) ? input.dims.slice(0).map((val, i) => i) : attributes.axes;
  const normalizedAxes = ShapeUtil.normalizeAxes(axes, input.dims.length);
  const starts = attributes.starts.map((start, i) => {
    if (start > input.dims[normalizedAxes[i]] - 1) {
      return input.dims[normalizedAxes[i]];
    }
    return ShapeUtil.normalizeAxis(start, input.dims[normalizedAxes[i]]);
  });
  const ends = attributes.ends.map((end, i) => {
    if (end > input.dims[normalizedAxes[i]] - 1) {
      return input.dims[normalizedAxes[i]];
    }
    return ShapeUtil.normalizeAxis(end, input.dims[normalizedAxes[i]]);
  });

  const outputShape = input.dims.slice();

  const sliceOps: string[] = [];
  for (let i = 0; i < normalizedAxes.length; i++) {
    outputShape[normalizedAxes[i]] = ends[i] - starts[i];
    if (starts[i] > 0) {
      sliceOps.push(`idx_${normalizedAxes[i]} += ${starts[i]}u;`);
    }  // else { sliceOps.push(`outputIdx[${normalizedAxes[i]}] += 0;`); }
  }

  const outputSize = ShapeUtil.size(outputShape);
  const outputStrides = ShapeUtil.computeStrides(outputShape);
  const shaderSource = `
  const WORKGROUP_SIZE: u32 = ${WORKGROUP_SIZE}u;
  @group(0) @binding(0) var<storage, read> input : array<${dataType}>;
  @group(0) @binding(1) var<storage, read_write> output : array<${dataType}>;

  @compute @workgroup_size(WORKGROUP_SIZE)
  fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {

    // Guard against out-of-bounds work group sizes
    if (global_id.x >= ${outputSize}u) {
      return;
    }

    var offset = global_id.x;
    ${offsetToIndices('offset', outputStrides, 'idx_')}
    ${sliceOps.join('')}
    var offsetInput = 0u;
    ${indicesToOffset('idx_', ShapeUtil.computeStrides(input.dims), 'offsetInput')}
    output[global_id.x] = input[offsetInput];
  }`;
  return {
    ...sliceProgramMetadata,
    outputs: [{dims: outputShape, type: input.type, gpuDataType: GpuDataType.default}],
    shaderSource,
    dispatchGroup: () => ({x: Math.ceil(outputSize / 64 /* workgroup size */)})
  };
};

const validateInputs = (inputs: Tensor[]): void => {
  if (!inputs || inputs.length !== 1) {
    throw new Error('Slice requires 1 input.');
  }
  if (NUMBER_TYPES.indexOf(inputs[0].type) === -1) {
    throw new Error('Invalid input type.');
  }
};

export const sliceV10 = async(inferenceHandler: WebGpuInferenceHandler, inputs: Tensor[]): Promise<Tensor[]> => {
  validateInputsV10(inputs);
  const attributes = generateSliceAttributesFromInputs(inferenceHandler, inputs);
  return inferenceHandler.run(
      {
        ...sliceProgramMetadata,
        cacheHint: attributes.cacheKey,
        get: () => createSliceProgramInfo(inputs[0], attributes)
      },
      [inputs[0]]);
};

const generateSliceAttributesFromInputs =
    (inferenceHandler: WebGpuInferenceHandler, inputs: Tensor[]): SliceAttributes => {
      if (!inferenceHandler.session.isInitializer(inputs[1].dataId) ||
          !inferenceHandler.session.isInitializer(inputs[2].dataId) ||
          (inputs.length >= 4 && !inferenceHandler.session.isInitializer(inputs[3].dataId)) ||
          (inputs.length >= 5 && !inferenceHandler.session.isInitializer(inputs[4].dataId))) {
        throw new Error('dynamic slice attributes are not allowed');
      }

      if (inputs.length >= 5 && inputs[4].integerData.some((i: number) => i !== 1)) {
        throw new Error('currently non-1 steps is not supported for Slice');
      }

      const starts = Array.from(inputs[1].integerData);
      const ends = Array.from(inputs[2].integerData);
      const axes = inputs.length >= 4 ? Array.from(inputs[3].integerData) : [];
      const cacheKey = `${axes};${starts};${ends}`;
      return {starts, ends, axes, cacheKey};
    };

const validateInputsV10 = (inputs: Tensor[]): void => {
  if (!inputs || inputs.length < 3 || inputs.length > 5) {
    throw new Error('Invalid input number.');
  }
  if (inputs[1].type !== 'int32' || inputs[1].dims.length !== 1) {
    throw new Error('Invalid input type.');
  }
  if (inputs[2].type !== 'int32' || inputs[2].dims.length !== 1) {
    throw new Error('Invalid input type.');
  }
  if (inputs.length >= 4 && (inputs[3].type !== 'int32' || inputs[3].dims.length !== 1)) {
    throw new Error('Invalid input type.');
  }
  if (inputs.length >= 5 && (inputs[4].type !== 'int32' || inputs[4].dims.length !== 1)) {
    throw new Error('Invalid input type.');
  }
};
