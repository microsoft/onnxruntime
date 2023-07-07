// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {DataType} from '../../../wasm-common';
import {TensorView} from '../../tensor';
import {PoolConvUtil} from '../../util';
import {AttributeWithCacheKey, createAttributeWithCacheKey} from '../attribute-with-cache-key';
import {ComputeContext} from '../types';

import {createGroupedConvProgramInfoLoader} from './conv-grouped';
import {createConv2DMatMulProgramInfoLoader} from './conv2d-mm';
import {InternalActivationAttributes, parseInternalActivationAttributes} from './fuse-utils';
import {createTransposeProgramInfo, TransposeAttributes, transposeProgramMetadata} from './transpose';

export const calculateOutputShape =
    (inputShape: readonly number[], kernelShape: readonly number[], dilations: readonly number[],
     adjustPads: readonly number[], strides: readonly number[], isChannelLast: boolean): number[] => {
      const batchSize = inputShape[0];
      const inputSpatialShape = inputShape.slice(isChannelLast ? 1 : 2, isChannelLast ? 3 : 4);
      const spatialRank = inputSpatialShape.length;
      const outChannels = kernelShape[0];
      const kernelSpatialShape = kernelShape.slice(2);
      const dilatedKernelShape = kernelSpatialShape.map((v, i) => v + (v - 1) * (dilations[i] - 1));
      const inputSpatialShapeWithPad = inputSpatialShape.map((v, i) => v + adjustPads[i] + adjustPads[i + spatialRank]);
      const outputShape =
          inputSpatialShapeWithPad.map((v, i) => Math.floor((v - dilatedKernelShape[i] + strides[i]) / strides[i]));
      outputShape.splice(0, 0, batchSize);
      outputShape.splice(isChannelLast ? 3 : 1, 0, outChannels);
      return outputShape;
    };

export interface ConvAttributes extends InternalActivationAttributes, AttributeWithCacheKey {
  readonly autoPad: string;
  readonly dilations: readonly number[];
  readonly format: 'NHWC'|'NCHW';
  readonly group: number;
  readonly kernelShape: readonly number[];
  readonly pads: readonly number[];
  readonly strides: readonly number[];
  readonly wIsConst: boolean;
}

// for transposing weight tensor from [M, C/group, KH, KW] to [KH, KW, C/group, M]
const weightTransposeAttribute: TransposeAttributes = createAttributeWithCacheKey({perm: [2, 3, 1, 0]});

const validateInputs = (inputs: readonly TensorView[], attributes: ConvAttributes): void => {
  // Refer to the below link for all input checks
  // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Conv
  if (!inputs || (inputs.length !== 2 && inputs.length !== 3)) {
    throw new Error('Conv requires 2 or 3 inputs');
  }

  // TODO : Need to add support for multi-dimensional conv
  if (inputs[0].dims.length !== 4 && inputs[0].dims.length !== 3) {
    throw new Error('currently only support conv 1D and 2D');
  }

  if (inputs[0].dims.length !== inputs[1].dims.length) {
    throw new Error('filter does not have same dimension as input');
  }

  // FILTER_IN_CHANNEL should be equal to DATA_CHANNEL
  const dataChannel = inputs[0].dims[attributes.format === 'NHWC' ? inputs[0].dims.length - 1 : 1];
  const filterInChannel = inputs[1].dims[1] * attributes.group;
  if (dataChannel !== filterInChannel) {
    throw new Error('FILTER_IN_CHANNEL should be equal to DATA_CHANNEL');
  }

  // if bias is provided it should be 1D and the number of elements should be equal to the number of feature maps
  if (inputs.length === 3 && (inputs[2].dims.length !== 1 || inputs[1].dims[0] !== inputs[2].dims[0])) {
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

  // if kernelShape is specified, it's data length must be 2 less than dims length of the weights tensor
  // (the first 2 dims are batch_size and channels)
  if (attributes.kernelShape.length !== 0 && attributes.kernelShape.length !== inputs[1].dims.length - 2) {
    throw new Error('invalid kernel shape');
  }

  // TODO : Need to add support for float64
  if (inputs[0].dataType !== DataType.float || inputs[1].dataType !== DataType.float) {
    throw new Error('Conv input(X,W) should be float tensor');
  }

  if (inputs.length === 3 && inputs[2].dataType !== DataType.float) {
    throw new Error('Conv input(bias) should be float tensor');
  }
};

const getAdjustedConvAttributes = <T extends ConvAttributes>(attributes: T, inputs: readonly TensorView[]): T => {
  const kernelShape = attributes.kernelShape.slice();
  // if kernelShape is not specified in the attributes of this op, infer it from the weight tensor dims
  for (let i = 2; i < inputs[1].dims.length; ++i) {
    if (kernelShape[i - 2] === 0) {
      kernelShape[i - 2] = inputs[1].dims[i];
    }
  }
  const pads = attributes.pads.slice();
  PoolConvUtil.adjustPadsBasedOnAutoPad(
      inputs[0].dims, attributes.strides, attributes.dilations, kernelShape, pads, attributes.format === 'NHWC',
      attributes.autoPad);

  // always return a new object so does not modify the original attributes
  const newAttributes: T = Object.assign({}, attributes);
  Object.assign(newAttributes, {kernelShape, pads, cacheKey: attributes.cacheKey});
  return newAttributes;
};

export const parseConvAttributes = (attributes: Record<string, unknown>): ConvAttributes => {
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

  return createAttributeWithCacheKey(
      {autoPad, format, dilations, group, kernelShape, pads, strides, wIsConst, ...activationAttributes});
};

const conv2d = (context: ComputeContext, inputs: readonly TensorView[], attributes: ConvAttributes): void => {
  const adjustedAttributes = getAdjustedConvAttributes(attributes, inputs);

  // check attributes

  const hasBias = inputs.length === 3;
  // const hasPreluActivationWeights = false; /* TODO: add support for prelu activation weights */
  const isChannelsLast = attributes.format === 'NHWC';

  // const batchSize = context.inputs[0].dims[0];
  const inputHeight = inputs[0].dims[isChannelsLast ? 1 : 2];
  const inputWidth = inputs[0].dims[isChannelsLast ? 2 : 3];
  const inputChannels = inputs[0].dims[isChannelsLast ? 3 : 1];
  const weightHeight = inputs[1].dims[2];
  const weightWidth = inputs[1].dims[3];

  const outputShape = calculateOutputShape(
      inputs[0].dims, inputs[1].dims, attributes.dilations, adjustedAttributes.pads, attributes.strides,
      isChannelsLast);
  const outHeight = outputShape[isChannelsLast ? 1 : 2];
  const outWidth = outputShape[isChannelsLast ? 2 : 3];
  const outChannels = outputShape[isChannelsLast ? 3 : 1];

  const sameSize =
      isChannelsLast && weightHeight === inputHeight && weightWidth === inputWidth && attributes.autoPad === 'VALID';
  if (sameSize ||
      (weightHeight === 1 && weightWidth === 1 && attributes.dilations[0] === 1 && attributes.dilations[1] === 1 &&
       attributes.strides[0] === 1 && attributes.strides[1] === 1 &&
       (attributes.autoPad === 'SAME_UPPER' || attributes.autoPad === 'SAME_LOWER' ||
        attributes.autoPad === 'VALID'))) {
    // TODO: implement conv2dByMatMul()
    context.compute(createGroupedConvProgramInfoLoader(inputs, adjustedAttributes));
    return;
  }

  if (!isChannelsLast || attributes.group !== 1) {
    context.compute(createGroupedConvProgramInfoLoader(inputs, adjustedAttributes));
    return;
  }

  // TODO: implement conv2dWithIm2Col()

  const dimAOuter = isChannelsLast ? outHeight * outWidth : outChannels;
  const dimBOuter = isChannelsLast ? outChannels : outHeight * outWidth;
  const dimInner = weightHeight * weightWidth * inputChannels;

  const sequentialAccessByThreads = /* backend.adapterInfo.isIntel() */ true;

  // STEP.1: transpose weight
  const transposedWeight = (context.kernelCustomData.wT as TensorView | undefined) ??
      context.compute(
          {
            ...transposeProgramMetadata,
            cacheHint: weightTransposeAttribute.cacheKey,
            get: () => createTransposeProgramInfo(inputs[1], weightTransposeAttribute.perm)
          },
          {inputs: [1], outputs: [attributes.wIsConst ? -2 : -1]})[0];
  if (attributes.wIsConst && !context.kernelCustomData.wT) {
    context.kernelCustomData.wT = transposedWeight;
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
      createConv2DMatMulProgramInfoLoader(
          convInputs, adjustedAttributes, outputShape, dimAOuter, dimBOuter, dimInner, hasBias,
          sequentialAccessByThreads),
      {inputs: convInputs});
};

const conv1d = (context: ComputeContext, attributes: ConvAttributes): void => {
  // extend the input to 2D by adding H dimension
  const isChannelLast = attributes.format === 'NHWC';
  const inputs = [
    context.inputs[0].reshape(
        isChannelLast ?
            // [N, W, C] -> [N, H=1, W, C]
            [context.inputs[0].dims[0], 1, context.inputs[0].dims[1], context.inputs[0].dims[2]] :
            // [N, C, W] -> [N, C, H=1, W]
            [context.inputs[0].dims[0], context.inputs[0].dims[1], 1, context.inputs[0].dims[2]]),
    //[FILTER_OUT_CHANNEL, FILTER_IN_CHANNEL, kW] -> [FILTER_OUT_CHANNEL, FILTER_IN_CHANNEL, kH=1, kW]
    context.inputs[1].reshape([context.inputs[1].dims[0], context.inputs[1].dims[1], 1, context.inputs[1].dims[2]])
  ];
  if (context.inputs.length === 3) {
    inputs.push(context.inputs[2]);
  }
  const pads = [0, attributes.pads[0], 0, attributes.pads[1]];
  const strides = [1].concat(attributes.strides);
  const dilations = [1].concat(attributes.dilations);
  const kernelShape = [1].concat(attributes.kernelShape);
  const adjustedAttributes = getAdjustedConvAttributes({...attributes, pads, strides, dilations, kernelShape}, inputs);
  context.compute(createGroupedConvProgramInfoLoader(
      inputs, adjustedAttributes,
      outputShape => isChannelLast ? [outputShape[0], outputShape[2], outputShape[3]] : []));
};

export const conv = (context: ComputeContext, attributes: ConvAttributes): void => {
  validateInputs(context.inputs, attributes);  // currently will fail if not conv1D/2D
  if (context.inputs[0].dims.length === 3) {
    conv1d(context, attributes);
  } else {
    conv2d(context, context.inputs, attributes);
  }
};
