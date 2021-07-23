// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {InferenceHandler} from '../../../backend';
import {Graph} from '../../../graph';
import {OperatorImplementation, OperatorInitialization} from '../../../operators';
import {Tensor} from '../../../tensor';
import {PoolConvUtil, ShapeUtil} from '../../../util';
import {getGlsl} from '../glsl-source';
import {WebGLInferenceHandler} from '../inference-handler';
import {ProgramInfo, TextureType} from '../types';

import {createUnpackedGroupedConvProgramInfo} from './conv-grouped';
import {conv2DPacked} from './conv-pack';
import {getActicationSnippet, parseInternalActivationAttributes} from './fuse-utils';
import {calculateIm2ColDims, createIm2ColProgramInfo} from './im2col';

function createDotProductProgramInfo(
    inferenceHandler: WebGLInferenceHandler, inputs: readonly Tensor[], outputShape: number[],
    attributes: ConvAttributes): ProgramInfo {
  const xshape = inputs[0].dims;
  const kshape = inputs[1].dims;
  const adjustedKernelShape = [kshape[0], Math.ceil((xshape[1] * kshape[2] * kshape[3]) / 4)];
  const im2colShape = calculateIm2ColDims(xshape, kshape, outputShape);
  const [kWidth, kHeight] =
      inferenceHandler.calculateTextureWidthAndHeight(adjustedKernelShape, TextureType.packedLastDimension);

  const im2colStrides = ShapeUtil.computeStrides(im2colShape);
  const [im2colWidth, im2colHeight] =
      inferenceHandler.calculateTextureWidthAndHeight(im2colShape, TextureType.packedLastDimension);
  const rank = outputShape.length;

  const initValue = (inputs.length < 3) ? '0.0' : '_B(b)';
  const sharedDim = Math.ceil(xshape[1] * kshape[2] * kshape[3] / 4);
  const {activationFunction, applyActivation} = getActicationSnippet(attributes);
  const glsl = getGlsl(inferenceHandler.session.backend.glContext.version);
  const shaderSource = `
  ${activationFunction}
  float process(int indices[${rank}]) {
    int b[1];
    b[0] = indices[1];
    int im2col[4];
    im2col[0] = indices[0];
    im2col[1] = indices[2];
    im2col[2] = indices[3];
    int im2colOffset = im2col[0] * ${im2colStrides[0]} + im2col[1] * ${im2colStrides[1]} + im2col[2] * ${
      im2colStrides[2]};
    int kernelOffset = indices[1] * ${adjustedKernelShape[1]};
    float value = ${initValue};
    for (int i = 0; i < ${sharedDim}; ++i) {
      vec2 im2colCoords = offsetToCoords(im2colOffset, ${im2colWidth}, ${im2colHeight});
      vec2 kernelCoords = offsetToCoords(kernelOffset, ${kWidth}, ${kHeight});
      value += dot(${glsl.texture2D}(Im2Col, im2colCoords), ${glsl.texture2D}(K, kernelCoords));
      ++im2colOffset;
      ++kernelOffset;
    }
    ${applyActivation}
    return value;
  }`;
  return {
    name: 'ConvDotProduct',
    inputNames: inputs.length === 3 ? ['Im2Col', 'K', 'B'] : ['Im2Col', 'K'],
    inputTypes: inputs.length === 3 ? [TextureType.unpacked, TextureType.packedLastDimension, TextureType.unpacked] :
                                      [TextureType.unpacked, TextureType.packedLastDimension],
    output: {dims: outputShape, type: inputs[0].type, textureType: TextureType.unpacked},
    shaderSource
  };
}

export const calculateOutputShape =
    (inputShape: readonly number[], kernelShape: readonly number[], dilations: readonly number[],
     adjustPads: readonly number[], strides: readonly number[]): number[] => {
      const batchSize = inputShape[0];
      const inputSpatialShape = inputShape.slice(2);
      const spatialRank = inputSpatialShape.length;
      const outChannels = kernelShape[0];
      const kernelSpatialShape = kernelShape.slice(2);
      const dilatedKernelShape = kernelSpatialShape.map((v, i) => v + (v - 1) * (dilations[i] - 1));
      const inputSpatialShapeWithPad = inputSpatialShape.map((v, i) => v + adjustPads[i] + adjustPads[i + spatialRank]);
      const outputSpatialShape =
          inputSpatialShapeWithPad.map((v, i) => Math.floor((v - dilatedKernelShape[i] + strides[i]) / strides[i]));
      const outputShape = [batchSize, outChannels].concat(...outputSpatialShape);
      return outputShape;
    };

export interface ConvAttributes {
  readonly autoPad: string;
  readonly dilations: readonly number[];
  readonly group: number;
  readonly kernelShape: readonly number[];
  readonly pads: readonly number[];
  readonly strides: readonly number[];
  readonly activation: string;
  readonly clipMax: number;
  readonly clipMin: number;
}

export const conv: OperatorImplementation<ConvAttributes> =
    (inferenceHandler: InferenceHandler, inputs: Tensor[], attributes: ConvAttributes): Tensor[] => {
      validateInputs(inputs, attributes);  // currently will fail if not conv2D
      return conv2d(inferenceHandler, inputs, attributes);
    };

const conv2d: OperatorImplementation<ConvAttributes> =
    (inferenceHandler: WebGLInferenceHandler, inputs: Tensor[], attributes: ConvAttributes): Tensor[] => {
      const adjustedAttributes = getAdjustedConvAttributes(attributes, inputs);
      const packMode = inferenceHandler.session.pack;
      const isPointwise = adjustedAttributes.kernelShape[0] === 1 && adjustedAttributes.kernelShape[1] === 1;
      if (adjustedAttributes.group > 1) {
        const result = inferenceHandler.run(
            createUnpackedGroupedConvProgramInfo(inferenceHandler, inputs, adjustedAttributes), inputs);
        return [result];
      } else if (packMode && inputs[0].dims.length === 4 && inputs[0].dims[0] === 1 && !isPointwise) {
        return [conv2DPacked(inferenceHandler, inputs, adjustedAttributes)];
      } else {
        return [conv2DUnpacked(inferenceHandler, inputs, adjustedAttributes)];
      }
    };

export const conv2DUnpacked =
    (inferenceHandler: WebGLInferenceHandler, inputs: readonly Tensor[], attributes: ConvAttributes): Tensor => {
      const xshape = inputs[0].dims;
      const kshape = inputs[1].dims;
      const outputShape =
          calculateOutputShape(xshape, kshape, attributes.dilations, attributes.pads, attributes.strides);
      const xIm2Col = inferenceHandler.run(
          createIm2ColProgramInfo(inferenceHandler, inputs[0], inputs[1], outputShape, attributes), [inputs[0]]);

      const dotProductInputs = inputs.length === 3 ? [xIm2Col, inputs[1], inputs[2]] : [xIm2Col, inputs[1]];
      const output = inferenceHandler.run(
          createDotProductProgramInfo(inferenceHandler, inputs, outputShape, attributes), dotProductInputs);
      return output;
    };

const getAdjustedConvAttributes = (attributes: ConvAttributes, inputs: Tensor[]): ConvAttributes => {
  const kernelShape = attributes.kernelShape.slice();
  // if kernelShape is not specified in the attributes of this op, infer it from the weight tensor dims
  if (attributes.kernelShape.length === 0) {
    for (let i = 2; i < inputs[1].dims.length; ++i) {
      kernelShape.push(inputs[1].dims[i]);
    }
  }
  const pads = attributes.pads.slice();
  PoolConvUtil.adjustPadsBasedOnAutoPad(
      inputs[0].dims, attributes.strides, attributes.dilations, kernelShape, pads, attributes.autoPad);

  // always return a new object so does not modify the original attributes
  const newAttributes: ConvAttributes = Object.assign({}, attributes);
  Object.assign(newAttributes, {kernelShape, pads});
  return newAttributes;
};

export const parseConvAttributes: OperatorInitialization<ConvAttributes> = (node: Graph.Node): ConvAttributes => {
  const attributes = node.attributes;
  // TODO : Make this generic enough to compute default attributes for multi-dimensional conv
  return {
    autoPad: attributes.getString('auto_pad', 'NOTSET'),
    dilations: attributes.getInts('dilations', [1, 1]),
    group: attributes.getInt('group', 1),
    kernelShape: attributes.getInts('kernel_shape', []),
    pads: attributes.getInts('pads', [0, 0, 0, 0]),
    strides: attributes.getInts('strides', [1, 1]),
    ...parseInternalActivationAttributes(attributes)
  };
};

const validateInputs = (inputs: Tensor[], attributes: ConvAttributes): void => {
  // Refer to the below link for all input checks
  // https://github.com/onnx/onnx/blob/master/docs/Operators.md#Conv
  if (!inputs || (inputs.length !== 2 && inputs.length !== 3)) {
    throw new Error('Conv requires 2 or 3 inputs');
  }

  // TODO : Need to add support for multi-dimensional conv
  if (inputs[0].dims.length !== 4 || inputs[1].dims.length !== 4) {
    throw new Error('currently only support 2-dimensional conv');
  }

  // FILTER_IN_CHANNEL should be equal to DATA_CHANNEL
  const dataChannel = inputs[0].dims[1];
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
  if (inputs[0].type !== 'float32' || inputs[1].type !== 'float32') {
    throw new Error('Conv input(X,W) should be float tensor');
  }

  if (inputs.length === 3 && inputs[2].type !== 'float32') {
    throw new Error('Conv input(bias) should be float tensor');
  }
};
