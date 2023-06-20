// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {DataType} from '../../../wasm-common';
import {TensorView} from '../../tensor';
import {createAttributeWithCacheKey} from '../attribute-with-cache-key';
import {ComputeContext} from '../types';

import {ConvAttributes} from './conv';
import {createConv2dTransposeMatMulProgramInfoLoader} from './conv-transpose-mm';
import {parseInternalActivationAttributes} from './fuse-utils';
import {createTransposeProgramInfo, TransposeAttributes, transposeProgramMetadata} from './transpose';

const computeTotalPad =
    (inDim: number, stride: number, adj: number, kernel: number, dilation: number, outSize: number) =>
        (inDim - 1) * stride + adj + (kernel - 1) * dilation + 1 - outSize;

const distributePadding = (totalPad: number, autoPad: string, pads: number[], head: number, tail: number) => {
  const smallPad = Math.floor(totalPad / 2);
  if (autoPad === 'SAME_UPPER') {
    pads[head] = smallPad;
    pads[tail] = totalPad - smallPad;
  } else if (autoPad === 'SAME_LOWER') {
    pads[head] = totalPad - smallPad;
    pads[tail] = smallPad;
  }
};

const calculateOutputShapeAndPads =
    (inputShape: readonly number[], kernelShape: readonly number[], dilations: readonly number[], autoPad: string,
     group: number, pads: number[], strides: readonly number[], isChannelLast: boolean, outputPadding: number[],
     outputShape: number[]) => {
      const spatialRank = inputShape.length - 2;
      const updateOutputShape = outputShape.length === 0;
      if (outputPadding.length === 0) {
        for (let i = 0; i < spatialRank; ++i) {
          outputPadding.push(0);
        }
      }
      const batchSize = inputShape[0];
      const outChannels = kernelShape[isChannelLast ? 3 : 1] * group;
      for (let i = 0, j = inputShape.length - spatialRank - (isChannelLast ? 1 : 0); i < spatialRank; ++i, ++j) {
        const inSize = inputShape[j];
        const outSize = updateOutputShape ? inSize * strides[i] : outputShape[i];
        const totalPad = computeTotalPad(inSize, strides[i], pads[i], kernelShape[j], dilations[i], outSize);
        distributePadding(totalPad, autoPad, pads, i, i + spatialRank);
        if (updateOutputShape) {
          outputShape.push(
              strides[i] * (inSize - 1) + outputPadding[i] + (kernelShape[j] - 1) * dilations[i] + 1 - pads[i] -
              pads[i + spatialRank]);
        }
      }
      outputShape.splice(0, 0, batchSize);
      outputShape.splice(isChannelLast ? 3 : 1, 0, outChannels);
    };

export interface ConvTransposeAttributes extends ConvAttributes {
  readonly outputPadding: readonly number[];
  readonly outputShape: readonly number[];
}

// for transposing weight tensor from [C, M/group, kH, kW] to [kH, kW, C, M/group]
const weightTransposeAttribute: TransposeAttributes = createAttributeWithCacheKey({perm: [2, 3, 0, 1]});

const getAdjustedConvTransposeAttributes =
    <T extends ConvTransposeAttributes>(attributes: T, inputs: readonly TensorView[]): T => {
      const kernelShape = attributes.kernelShape.slice();
      // if kernelShape is not specified in the attributes of this op, infer it from the weight tensor dims
      if (attributes.kernelShape.length === 0 || attributes.kernelShape.reduce((a, b) => a * b, 0) === 0) {
        kernelShape.length = 0;
        for (let i = 2; i < inputs[1].dims.length; ++i) {
          kernelShape.push(inputs[1].dims[i]);
        }
      }
      const isChannelsLast = attributes.format === 'NHWC';
      kernelShape.splice(0, 0, inputs[1].dims[0]);
      kernelShape.splice(isChannelsLast ? 3 : 1, 0, inputs[1].dims[1]);

      const pads = attributes.pads.slice();
      const outputShape = attributes.outputShape.slice();
      const outputPadding = attributes.outputPadding.slice();
      const inputShape = inputs[0].dims;
      // If outputShape is not specified in the attributes of this op, infer it from the parameters
      // Similarly, automatically infer pads if not specified
      calculateOutputShapeAndPads(
          inputShape, kernelShape, attributes.dilations, attributes.autoPad, attributes.group, pads, attributes.strides,
          isChannelsLast, outputPadding, outputShape);

      // always return a new object so does not modify the original attributes
      const newAttributes: T = Object.assign({}, attributes);
      Object.assign(newAttributes, {kernelShape, pads, outputPadding, outputShape, cacheKey: attributes.cacheKey});
      return newAttributes;
    };

export const parseConvTransposeAttributes = (attributes: Record<string, unknown>): ConvTransposeAttributes => {
  const activationAttributes = parseInternalActivationAttributes(attributes);
  // TODO : Make this generic enough to compute default attributes for multi-dimensional conv
  const format = attributes.format as 'NHWC' | 'NCHW';
  const autoPad = ['NOTSET', 'VALID', 'SAME_UPPER', 'SAME_LOWER'][attributes.auto_pad as number];
  const dilations = attributes.dilations as [number, number];
  const group = attributes.group as number;
  const kernelShape = attributes.kernelShape as [number, number];
  const pads = attributes.pads as [number, number, number, number];
  const strides = attributes.strides as [number, number];
  const wIsConst = (attributes.wIsConst as () => boolean)();
  const outputPadding = attributes.outputPadding as [number, number, number, number];
  const outputShape = attributes.outputShape as [number, number];
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

  if (inputs[0].dims.length !== inputs[1].dims.length) {
    throw new Error('filter does not have same dimension as input');
  }

  // FILTER_IN_CHANNEL should be equal to DATA_CHANNEL
  const dataChannel = attributes.format === 'NHWC' ? inputs[0].dims[3] : inputs[0].dims[1];
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
  if (attributes.outputPadding.length !== spatialRank && attributes.outputPadding.length !== 0) {
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


const convTranspose2d =
    (context: ComputeContext, inputs: readonly TensorView[], attributes: ConvTransposeAttributes): void => {
      const adjustedAttributes = getAdjustedConvTransposeAttributes(attributes, inputs);

      const hasBias = inputs.length === 3;
      // const hasPreluActivationWeights = false; /* TODO: add support for prelu activation weights */
      const isChannelsLast = adjustedAttributes.format === 'NHWC';

      // const batchSize = context.inputs[0].dims[0];
      // const inputHeight = inputs[0].dims[isChannelsLast ? 1 : 2];
      // const inputWidth = inputs[0].dims[isChannelsLast ? 2 : 3];
      const inputChannels = inputs[0].dims[isChannelsLast ? 3 : 1];
      const weightHeight = inputs[1].dims[2];
      const weightWidth = inputs[1].dims[3];

      const outputShape = adjustedAttributes.outputShape;
      const outHeight = outputShape[isChannelsLast ? 1 : 2];
      const outWidth = outputShape[isChannelsLast ? 2 : 3];
      const outChannels = outputShape[isChannelsLast ? 3 : 1];

      const dimAOuter = isChannelsLast ? outHeight * outWidth : outChannels;
      const dimBOuter = isChannelsLast ? outChannels : outHeight * outWidth;
      const dimInner = weightHeight * weightWidth * inputChannels;

      const sequentialAccessByThreads = /* backend.adapterInfo.isIntel() */ true;

      // STEP.1: transpose weight
      const transposedWeight = (context.customData.wT as TensorView | undefined) ??
          context.compute(
              {
                ...transposeProgramMetadata,
                cacheHint: weightTransposeAttribute.cacheKey,
                get: () => createTransposeProgramInfo(inputs[1], weightTransposeAttribute.perm)
              },
              {inputs: [1], outputs: [attributes.wIsConst ? -2 : -1]})[0];
      if (attributes.wIsConst && !context.customData.wT) {
        context.customData.wT = transposedWeight;
      }

      // STEP.2: prepare reshaped inputs
      const convInputs = [inputs[0], transposedWeight];
      if (hasBias) {
        if (!isChannelsLast && inputs[2].dims.length === 1) {
          convInputs.push(inputs[2].reshape([inputs[2].dims[0], 1, 1]));
        } else {
          convInputs.push(inputs[2]);
        }
      }

      // STEP.3: compute matmul
      context.compute(
          createConv2dTransposeMatMulProgramInfoLoader(
              convInputs, adjustedAttributes, outputShape, dimAOuter, dimBOuter, dimInner, hasBias,
              sequentialAccessByThreads),
          {inputs: convInputs});
    };

export const convTranspose = (context: ComputeContext, attributes: ConvTransposeAttributes): void => {
  validateInputs(context.inputs, attributes);
  convTranspose2d(context, context.inputs, attributes);
};
