// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {DataType} from '../../../wasm-common';
import {TensorView} from '../../tensor-view';
import {ShapeUtil} from '../../util';
import {AttributeWithCacheKey, createAttributeWithCacheKey} from '../attribute-with-cache-key';
import {ComputeContext, ProgramInfo, TensorInfo} from '../types';

import {IndicesHelper, inputVariable, outputVariable, ShaderHelper} from './common';

export interface SliceAttributes extends AttributeWithCacheKey {
  readonly starts: number[];
  readonly ends: number[];
  readonly axes: number[];
}

const validateInputs = (inputs: readonly TensorView[], attributes: SliceAttributes): void => {
  if (!inputs || inputs.length < 1) {
    throw new Error('too few inputs');
  }
  if (attributes.axes.length !== 0) {
    if (attributes.axes.length !== attributes.starts.length || attributes.axes.length !== attributes.ends.length) {
      throw new Error('axes, starts and ends must have the same length');
    }
  } else if (attributes.starts.length !== attributes.ends.length) {
    throw new Error('starts and ends must have the same length');
  }
  inputs.slice(1).forEach((_, idx) => {
    if (inputs[idx + 1].dataType !== DataType.int32 && inputs[idx + 1].dataType !== DataType.int64) {
      throw new Error(`Input ${idx} must be an array of int32 or int64`);
    }
  });
};

const readInput = (inputs: readonly TensorView[], idx: number): number[] => {
  const input: number[] = [];
  if (inputs.length > idx) {
    if (inputs[idx].dataType === DataType.int64) {
      inputs[idx].getBigInt64Array().forEach(v => input.push(Number(v)));
    } else if (inputs[idx].dataType === DataType.int32) {
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

const fixStartEndValues =
    (value: number, index: number, inputShape: readonly number[], axes: readonly number[], steps: readonly number[]):
        number => {
          let newValue = value;
          if (value < 0) {
            newValue += inputShape[axes[index]];
          }
          if (steps[index] < 0) {
            return Math.max(0, Math.min(newValue, inputShape[axes[index]] - 1));
          } else {
            return Math.max(0, Math.min(newValue, inputShape[axes[index]]));
          }
        };

const calculateInputIndicesImpl =
    (input: IndicesHelper, output: IndicesHelper, inputShape: readonly number[], outputShape: readonly number[]):
        string => `fn calculateInputIndices(outputIndices: ${output.type.indices}) -> ${input.type.indices} {
          var inputIndices: ${input.type.indices};
          var carry = 0u;
          for (var i = ${inputShape.length}; i >= 0; i--) {
            var outputIndex = ${outputShape.length === 1 ? 'outputIndices' : 'outputIndices[i]'};
            var inputIndex = outputIndex * steps[i] + starts[i] + carry;
            carry = inputIndex / inputShape[i];
            inputIndex = inputIndex % inputShape[i];
            if (signs[i] < 0) {
              inputIndex = inputShape[i] - inputIndex - 1u + starts[i];
            }
            ${inputShape.length === 1 ? 'inputIndices' : 'inputIndices[i]'} = inputIndex;
          }
          return inputIndices;
      }`;

const createSliceProgramInfo = (inputs: readonly TensorView[], attributes: SliceAttributes): ProgramInfo => {
  const inputShape = inputs[0].dims;
  const inputSize = ShapeUtil.size(inputShape);
  const axes = (attributes.axes.length > 0) ? ShapeUtil.normalizeAxes(attributes.axes, inputShape.length) :
                                              [...Array(inputShape.length).keys()];
  let steps = readInput(inputs, 4);
  steps.forEach((step) => step !== 0 || (() => {
                            throw new Error('step cannot be 0');
                          }));
  if (steps.length === 0) {
    steps = Array(axes.length).fill(1);
  }
  const starts = attributes.starts.map((start, i) => fixStartEndValues(start, i, inputShape, axes, steps));

  const ends = attributes.ends.map((end, i) => fixStartEndValues(end, i, inputShape, axes, steps));

  if (axes.length !== inputShape.length) {
    for (let i = 0; i < inputShape.length; ++i) {
      if (!axes.includes(i)) {
        starts.splice(i, 0, 0);
        ends.splice(i, 0, inputShape[i]);
        steps.splice(i, 0, 1);
      }
    }
  }
  const signs = steps.map(step => Math.sign(step));
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

  const outputShape = inputShape.slice(0);
  axes.forEach((axis, _) => {
    outputShape[axis] = Math.ceil((ends[axis] - starts[axis]) / steps[axis]);
  });

  const outputTensorInfo: TensorInfo = {dims: outputShape, dataType: inputs[0].dataType};

  const output = outputVariable('output', inputs[0].dataType, outputShape);
  const input = inputVariable('input', inputs[0].dataType, inputShape);
  const outputSize = ShapeUtil.size(outputShape);

  const getShaderSource = (shaderHelper: ShaderHelper) => `
      ${shaderHelper.declareVariables(input, output)}
        const signs = array<i32, ${signs.length}>(${signs.map(i => `${i}i`).join(',')});
        const starts = array<u32, ${starts.length}>(${starts.map(i => `${i}u`).join(',')});
        const ends = array<u32, ${ends.length}>(${ends.map(i => `${i}u`).join(',')});
        const steps = array<u32, ${steps.length}>(${steps.map(i => `${i}u`).join(',')});
        const inputShape = array<u32, ${inputShape.length}>(${inputShape.map(i => `${i}u`).join(',')});

        ${calculateInputIndicesImpl(input, output, inputShape, outputShape)}
        ${shaderHelper.mainStart()}
          ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes(outputSize)}
          let outputIndices = ${output.offsetToIndices('global_idx')};
          let inputIndices = calculateInputIndices(outputIndices);
          ${output.setByOffset('global_idx', input.getByIndices('inputIndices'))}
      }`;
  return {
    name: 'Slice',
    shaderCache: {hint: `${attributes.cacheKey}|${inputs[4]?.dims ?? ''}`},
    getShaderSource,
    getRunData: () => ({
      outputs: [outputTensorInfo],
      dispatchGroup: {x: Math.ceil(inputSize / 64 /* workgroup size */)},
    })
  };
};

export const slice = (context: ComputeContext, attributes: SliceAttributes): void => {
  validateInputs(context.inputs, attributes);
  const updatedAttributes = createSliceAttributesFromInputs(context.inputs, attributes);
  context.compute(createSliceProgramInfo(context.inputs, updatedAttributes), {inputs: [0]});
  // if (ShapeUtil.size(program.outputs[0].dims) > 0) {
  //   context.compute(programInfoLoader, {inputs: [0]});
  // } else {
  //   // TODO: support empty output
  //   throw new Error('slice: output size is 0');
  // }
};

export const parseSliceAttributes = (attributes: Record<string, unknown>): SliceAttributes => {
  const starts = attributes.starts as number[];
  const ends = attributes.ends as number[];
  const axes = attributes.axes as number[];
  return createAttributeWithCacheKey({starts, ends, axes});
};
