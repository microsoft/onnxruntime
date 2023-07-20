// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {DataType} from '../../../wasm-common';
import {TensorView} from '../../tensor';
import {ShapeUtil} from '../../util';
import {AttributeWithCacheKey, createAttributeWithCacheKey} from '../attribute-with-cache-key';
import {ComputeContext, GpuDataType, ProgramInfo, ProgramInfoLoader, ProgramMetadata, TensorInfo} from '../types';

import {createIndicesHelper, ShaderHelper} from './common';

export interface SliceAttributes extends AttributeWithCacheKey {
  readonly starts: number[];
  readonly ends: number[];
  readonly axes: number[];
}

const validateInputs = (inputs: readonly TensorView[], attributes: SliceAttributes): void => {
  if (!inputs || inputs.length < 1) {
    throw new Error('too few inputs');
  }
  if (attributes.axes.length !== attributes.starts.length || attributes.axes.length !== attributes.ends.length) {
    throw new Error('axes, starts and ends must have the same length');
  }
  inputs.slice(1).forEach((_, idx) => {
    if (inputs[idx + 1].dataType !== DataType.int32 && inputs[idx + 1].dataType !== DataType.int64) {
      throw new Error(`Input ${idx} must be an array of int32 or int64`);
    }
  });
  if (inputs.length > 4) {
    if (inputs[4].dataType === DataType.int64) {
      inputs[4].getBigInt64Array().forEach(v => {
        if (Number(v) === 0) {
          throw new Error('step cannot be 0');
        }
      });
    } else if (inputs[4].dataType !== DataType.int32) {
      inputs[4].getInt32Array().forEach(v => {
        if (Number(v) === 0) {
          throw new Error('step cannot be 0');
        }
      });
    }
  }
};

const readInput = (inputs: readonly TensorView[], idx: number): number[] => {
  const input: number[] = [];
  if (inputs.length > idx) {
    if (inputs[idx].dataType === DataType.int64) {
      inputs[idx].getBigInt64Array().forEach(v => input.push(Number(v)));
    } else if (inputs[1].dataType === DataType.int32) {
      inputs[idx].getInt32Array().forEach(v => input.push(Number(v)));
    } else {
      throw new Error(`Input ${idx} must be an array of int32 or int64`);
    }
  }
  return input;
};

const createSliceAttributesFromInputs =
    (inputs: readonly TensorView[], attributes: SliceAttributes): SliceAttributes => {
      if (inputs.length > 1) {
        const starts: number[] = readInput(inputs, 1);
        const ends: number[] = readInput(inputs, 2);
        let axes: number[] = readInput(inputs, 3);
        if (axes.length === 0) {
          axes = [...Array(inputs[0].dims.length).keys()];
        }
        return createAttributeWithCacheKey({starts, ends, axes});
      } else {
        return attributes;
      }
    };

const createSliceProgramInfo =
    (metadata: ProgramMetadata, inputs: readonly TensorView[], attributes: SliceAttributes): ProgramInfo => {
      const inputShape = inputs[0].dims;
      const inputSize = ShapeUtil.size(inputShape);
      const axes = ShapeUtil.normalizeAxes(attributes.axes, inputShape.length);
      const dataType = 'f32';  // TODO: support other data type
      let steps = readInput(inputs, 4);
      if (steps.length === 0) {
        steps = Array(axes.length).fill(1);
      }
      const starts = attributes.starts.map((start, i) => {
        let newStart = start;
        if (start < 0) {
          newStart += inputShape[axes[i]];
        }
        if (steps[i] < 0) {
          return Math.max(0, Math.min(newStart, inputShape[axes[i]] - 1));
        } else {
          return Math.max(0, Math.min(newStart, inputShape[axes[i]]));
        }
      });

      const ends = attributes.ends.map((end, i) => {
        let newEnd = end;
        if (end < 0) {
          newEnd = end + inputShape[axes[i]];
        }
        if (steps[i] < 0) {
          return Math.max(-1, Math.min(newEnd, inputShape[axes[i]] - 1));
        } else {
          return Math.max(0, Math.min(newEnd, inputShape[axes[i]]));
        }
      });

      const outputShape = inputShape.slice(0);
      axes.forEach((axis, i) => {
        outputShape[axis] = Math.ceil((ends[i] - starts[i]) / steps[i]);
      });
      if (axes.length !== inputShape.length) {
        for (let i = 0; i < inputShape.length; ++i) {
          if (!axes.includes(i)) {
            starts.splice(i, 0, 0);
            ends.splice(i, 0, inputShape[i]);
            steps.splice(i, 0, 1);
          }
        }
      }
      let signs = steps.map(step => Math.sign(step));
      // Convert negative steps to positive steps and reverse starts and ends
      steps.forEach((step, i, array) => {
        if (step < 0) {
          const numSteps = (ends[i] - starts[i]) / step;
          const newEnd = starts[i];
          const newStart = newEnd + numSteps * steps[i];
          starts[i] = newStart;
          ends[i] = newEnd;
          array[i] = -step;
        }
      });

      const output: TensorInfo = {dims: outputShape, dataType: inputs[0].dataType, gpuDataType: GpuDataType.default};

      const outputIndicesHelper = createIndicesHelper('output', outputShape);
      const inputIndicesHelper = createIndicesHelper('input', inputShape);
      const outputSize = ShapeUtil.size(outputShape);
      const calculateInputIndicesImpl = (): string => `
        fn calculateInputIndices(outputIndices: ${outputIndicesHelper.iType}) -> ${inputIndicesHelper.iType} {
          ${inputIndicesHelper.indicesVariableDeclaration('inputIndices', Array(inputShape.length).fill('0u'))};
          var carry = 0u;
          for (var i = ${inputShape.length}; i >= 0; i--) {
            if (i < ${outputShape.length}) {
              var n = outputIndices[i] * steps[i] + starts[i] + carry;
              carry = n / inputShape[i];
              n = n % inputShape[i];
              inputIndices[i] = abs(n);
            } else {
              inputIndices[i] = carry % inputShape[i];
              carry /= inputShape[i];
            }
            if (signs[i] < 0) {
              inputIndices[i] = inputShape[i] - inputIndices[i] - 1;
            }
          }
          return inputIndices;
      }`;
      const getShaderSource = (shaderHelper: ShaderHelper) => `
        @group(0) @binding(0) var<storage, read> input: array<${dataType}>;
        @group(0) @binding(1) var<storage, read_write> output: array<${dataType}>;
        const signs = array<i32, ${axes.length}>(${signs.map(i => `${i}i`).join(',')});
        const starts = array<u32, ${starts.length}>(${starts.map(i => `${i}u`).join(',')});
        const ends = array<u32, ${ends.length}>(${ends.map(i => `${i}u`).join(',')});
        const steps = array<u32, ${steps.length}>(${steps.map(i => `${i}u`).join(',')});
        const inputShape = array<u32, ${inputShape.length}>(${inputShape.map(i => `${i}u`).join(',')});

        ${outputIndicesHelper.o2iImpl}
        ${inputIndicesHelper.i2oImpl}
        ${calculateInputIndicesImpl()}
        ${shaderHelper.mainStart()}
          ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes(outputSize)}
          ${outputIndicesHelper.indicesVariableDeclaration('outputIndices')}
          ${outputIndicesHelper.o2iCall('global_idx', 'outputIndices')}
          ${inputIndicesHelper.indicesVariableDeclaration('inputIndices')}
          inputIndices = calculateInputIndices(outputIndices);
          output[global_idx] = input[${inputIndicesHelper.i2oExpression('inputIndices')}];
      }`;
      return {
        ...metadata,
        getShaderSource,
        outputs: [output],
        dispatchGroup: () => ({x: Math.ceil(inputSize / 64 /* workgroup size */)})
      };
    };

const createSliceProgramInfoLoader =
    (inputs: readonly TensorView[], attributes: SliceAttributes): ProgramInfoLoader => {
      const updatedAttributes = createSliceAttributesFromInputs(inputs, attributes);
      const metadata: ProgramMetadata = {
        name: 'Slice',
        inputTypes: [GpuDataType.default],
        cacheHint: updatedAttributes.cacheKey + (inputs.length > 4 ? 'steps_' + inputs[4].dims.toString() : '')
      };
      return {...metadata, get: () => createSliceProgramInfo(metadata, inputs, updatedAttributes)};
    };

export const slice = (context: ComputeContext, attributes: SliceAttributes): void => {
  validateInputs(context.inputs, attributes);
  context.compute(createSliceProgramInfoLoader(context.inputs, attributes), {inputs: [0]});
};
export const parseSliceAttributes = (attributes: Record<string, unknown>): SliceAttributes => {
  const starts = attributes.starts as number[];
  const ends = attributes.ends as number[];
  const axes = attributes.axes as number[];
  const steps: number[] = [];
  return createAttributeWithCacheKey({starts, ends, axes, steps});
};
