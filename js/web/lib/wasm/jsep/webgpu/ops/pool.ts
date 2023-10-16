// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {TensorView} from '../../tensor-view';
import {PoolConvUtil, ShapeUtil} from '../../util';
import {AttributeWithCacheKey, createAttributeWithCacheKey} from '../attribute-with-cache-key';
import {ComputeContext, ProgramInfo} from '../types';

import {IndicesHelper, inputVariable, outputVariable, ShaderHelper} from './common';

// TODO: support:
// - ceil_mode                 "test_maxpool_2d_ceil"
// - storage_order             "test_maxpool_with_argmax_2d_precomputed_strides"
// - [MaxPool] dilations       "test_maxpool_2d_dilations"
// - [MaxPool] output[1]       "test_maxpool_with_argmax_2d_precomputed_pads"

const validateInputs = (inputs: readonly TensorView[]): void => {
  if (!inputs || inputs.length !== 1) {
    throw new Error('Pool ops requires 1 input.');
  }
  if (inputs[0].dims.length !== 4 && inputs[0].dims.length !== 3) {
    throw new Error('Pool ops supports 1-D or 2-D inputs only for now.');
  }
};

const getAdjustedPoolAttributesAndOutputShape = <AttributeType extends AveragePoolAttributes|MaxPoolAttributes>(
    input: TensorView, attributes: AttributeType, isGlobalOperator: boolean): [AttributeType, number[]] => {
  const isChannelsLast = attributes.format === 'NHWC';
  const inputShapeAsChannelFirst = input.dims.slice();
  if (isChannelsLast) {
    inputShapeAsChannelFirst.splice(1, 0, inputShapeAsChannelFirst.pop()!);  // Move channel to the second position.
  }
  const hasDilations = Object.hasOwnProperty.call(attributes, 'dilations');
  const kernelShape = attributes.kernelShape.slice();
  const strides = attributes.strides.slice();
  const dilations: number[] = hasDilations ? (attributes as MaxPoolAttributes).dilations.slice() : [];
  const pads = attributes.pads.slice();
  PoolConvUtil.adjustPoolAttributes(isGlobalOperator, inputShapeAsChannelFirst, kernelShape, strides, dilations, pads);

  const outputShapeAsChannelFirst = PoolConvUtil.computePoolOutputShape(
      isGlobalOperator, inputShapeAsChannelFirst, strides, dilations, kernelShape, pads, attributes.autoPad);

  const newAttributes = Object.assign({}, attributes);
  if (hasDilations) {
    Object.assign(newAttributes, {kernelShape, strides, pads, dilations, cacheKey: attributes.cacheKey});
  } else {
    Object.assign(newAttributes, {kernelShape, strides, pads, cacheKey: attributes.cacheKey});
  }
  const outputShapeAsChannelLast = outputShapeAsChannelFirst.slice();
  outputShapeAsChannelLast.splice(1, 0, outputShapeAsChannelLast.pop()!);
  return [newAttributes, isChannelsLast ? outputShapeAsChannelLast : outputShapeAsChannelFirst];
};

const generatePoolingCode = <AttributeType extends AveragePoolAttributes|MaxPoolAttributes>(
    shaderHelper: ShaderHelper, x: IndicesHelper, xShape: readonly number[], outputShape: readonly number[],
    attributes: AttributeType, op1: string, op2: string, start: string): string => {
  const isChannelsLast = attributes.format === 'NHWC';
  const inputDims = xShape;
  const dataType = x.type.value;
  const rank = inputDims.length;
  const outputSize = ShapeUtil.size(outputShape);
  const output = outputVariable('output', x.type.tensor, outputShape);

  if (attributes.kernelShape.length <= 2) {
    const kw = attributes.kernelShape[attributes.kernelShape.length - 1];
    const sw = attributes.strides[attributes.strides.length - 1];
    const pwStart = attributes.pads[attributes.pads.length / 2 - 1];
    const pwEnd = attributes.pads[attributes.pads.length - 1];
    const dimIdxW = rank - (isChannelsLast ? 2 : 1);
    let codeW = '';
    let codeH = '';
    let codeHEnd = '';
    if (pwStart + pwEnd !== 0) {
      codeW = `
                for (var i: u32 = 0u; i < ${kw}u; i++) {
                  xIndices[${dimIdxW}] = indices[${dimIdxW}] * ${sw} - ${pwStart} + i;
                  if (xIndices[${dimIdxW}] < 0 || xIndices[${dimIdxW}] >= ${inputDims[dimIdxW]}) {
                    pad++;
                    continue;
                  }
                  let x_val = x[${x.indicesToOffset('xIndices')}];
                  ${op1}
                }`;
    } else {
      codeW = `
                for (var i: u32 = 0u; i < ${kw}u; i++) {
                  xIndices[${dimIdxW}] = indices[${dimIdxW}] * ${sw} - ${pwStart} + i;
                  let x_val = x[${x.indicesToOffset('xIndices')}];
                  ${op1}
                }`;
    }

    if (attributes.kernelShape.length === 2) {
      const kh = attributes.kernelShape[attributes.kernelShape.length - 2];
      const sh = attributes.strides[attributes.strides.length - 2];
      const phStart = attributes.pads[attributes.pads.length / 2 - 2];
      const phEnd = attributes.pads[attributes.pads.length - 2];
      const dimIdxH = rank - (isChannelsLast ? 3 : 2);
      const dimH = inputDims[dimIdxH];
      if (phStart + phEnd !== 0) {
        codeH = `
                for (var j: u32 = 0u; j < ${kh}u; j++) {
                  xIndices[${dimIdxH}] = indices[${dimIdxH}] * ${sh} - ${phStart} + j;
                  if (xIndices[${dimIdxH}] < 0 || xIndices[${dimIdxH}] >= ${dimH}) {
                    pad+= ${kw};
                    continue;
                  }
              `;
      } else {
        codeH = `
                for (var j: u32 = 0u; j < ${kh}u; j++) {
                  xIndices[${dimIdxH}] = indices[${dimIdxH}] * ${sh} - ${phStart} + j;
                `;
      }
      codeHEnd = `
              }
            `;
    }

    const poolingCode = `
            ${shaderHelper.declareVariables(x, output)}

            ${shaderHelper.mainStart()}
              ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes(outputSize)}

              let indices = ${output.offsetToIndices('global_idx')};
              var xIndices = ${output.offsetToIndices('global_idx')};

              var value: ${dataType} = ${dataType}(${start});
              var pad = 0;
              ${codeH}
              ${codeW}
              ${codeHEnd}
              ${op2}

              output[global_idx] = value;
            }`;
    return poolingCode;
  } else {
    if (isChannelsLast) {
      throw new Error('Pooling with kernelShape.length > 2 is not supported for NHWC format.');
    }
    const kernelSize = ShapeUtil.size(attributes.kernelShape);
    const kernelStrides = ShapeUtil.computeStrides(attributes.kernelShape);
    const stridesRank = kernelStrides.length;
    const padsRank = attributes.pads.length;
    const hasPads = attributes.pads.reduce((sum, cur) => sum + cur);
    let padCode = '';
    if (hasPads) {
      padCode = `
                if (xIndices[j] >= inputDims[j]) {
                  pad++;
                  isPad = true;
                  break;
                }
              }
              if (!isPad) {
                let x_val = x[${x.indicesToOffset('xIndices')}];
                ${op1}
              }`;
    } else {
      padCode = `
              }
              let x_val = x[${x.indicesToOffset('xIndices')}];
              ${op1}
            `;
    }
    const poolingCode = `
            ${shaderHelper.declareVariables(x, output)}

            const pads = array<u32, ${padsRank}>(${attributes.pads.map(i => `${i}u`).join(',')});
            const inputDims = array<u32, ${rank}>(${inputDims.map(i => `${i}u`).join(',')});
            const kernelStrides = array<u32, ${stridesRank}>(${kernelStrides.map(i => `${i}u`).join(',')});
            const strides = array<u32, ${stridesRank}>(${attributes.strides.map(i => `${i}u`).join(',')});

            ${shaderHelper.mainStart()}
              ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes(outputSize)}

              let indices = ${output.offsetToIndices('global_idx')};
              let xIndices = ${output.offsetToIndices('global_idx')};

              var offsets: array<u32, ${stridesRank}>;

              var value = ${output.type.value}(${start});
              var pad = 0;
              var isPad = false;

              for (var i: u32 = 0u; i < ${kernelSize}u; i++) {
                var offset = i;
                for (var j = 0u; j < ${stridesRank - 1}u; j++) {
                  offsets[j] = offset / kernelStrides[j];
                  offset -= offsets[j] * kernelStrides[j];
                }
                offsets[${stridesRank - 1}] = offset;

                isPad = false;
                for (var j = ${rank - stridesRank}u; j < ${rank}u; j++) {
                  xIndices[j] = indices[j] * strides[j - ${rank - stridesRank}u]
                    + offsets[j - ${rank - stridesRank}u] - pads[j - 2u];
                  ${padCode}
              }
              ${op2}

              output[global_idx] = value;
            }`;
    return poolingCode;
  }
};

export interface FormatAttributes {
  readonly format: 'NHWC'|'NCHW';
}

export interface PoolCommonAttributes extends FormatAttributes {
  readonly autoPad: string;
  readonly ceilMode: number;
  readonly kernelShape: readonly number[];
  readonly strides: readonly number[];
  readonly pads: readonly number[];
}

const parsePoolCommonAttributes = (attributes: Record<string, unknown>): PoolCommonAttributes => ({
  format: attributes.format as FormatAttributes['format'],
  autoPad: ['NOTSET', 'VALID', 'SAME_UPPER', 'SAME_LOWER'][attributes.auto_pad as number],
  ceilMode: attributes.ceil_mode as number,
  kernelShape: attributes.kernel_shape as [number, number],
  strides: attributes.strides as [number, number],
  pads: attributes.pads as [number, number, number, number]
});

export interface AveragePoolAttributes extends PoolCommonAttributes, AttributeWithCacheKey {
  readonly countIncludePad: boolean;
}

const createAveragePoolProgramInfo =
    (name: string, input: TensorView, isGlobalOperator: boolean, attributes: AveragePoolAttributes): ProgramInfo => {
      const [adjustedAttributes, outputShape] =
          getAdjustedPoolAttributesAndOutputShape(input, attributes, isGlobalOperator);
      const kernelSize = ShapeUtil.size(adjustedAttributes.kernelShape);

      const x = inputVariable('x', input.dataType, input.dims);
      const dataType = x.type.value;

      const op1 = 'value += x_val;';
      let op2 = '';
      if (adjustedAttributes.countIncludePad) {
        op2 += `value /= ${dataType}(${kernelSize});`;
      } else {
        op2 += `value /= ${dataType}(${kernelSize} - pad);`;
      }
      return {
        name,
        shaderCache: {hint: attributes.cacheKey},
        getRunData: () => ({
          outputs: [{dims: outputShape, dataType: input.dataType}],
          dispatchGroup: {x: Math.ceil(ShapeUtil.size(outputShape) / 64 /* workgroup size */)}
        }),
        getShaderSource: shaderHelper =>
            generatePoolingCode(shaderHelper, x, input.dims, outputShape, adjustedAttributes, op1, op2, '0.0'),
      };
    };

export const parseAveragePoolAttributes = (attributes: Record<string, unknown>): AveragePoolAttributes => {
  const countIncludePad = (attributes.count_include_pad as number) === 0 ? false : true;

  const attr = parsePoolCommonAttributes(attributes);
  // TODO: support attribute 'ceil_mode'
  if (attr.ceilMode !== 0) {
    throw new Error('using ceil() in shape computation is not yet supported for AveragePool');
  }

  return createAttributeWithCacheKey({countIncludePad, ...attr});
};

export const averagePool = (context: ComputeContext, attributes: AveragePoolAttributes): void => {
  validateInputs(context.inputs);
  context.compute(createAveragePoolProgramInfo('AveragePool', context.inputs[0], false, attributes));
};

const globalPoolAttributes = {
  autoPad: '',
  ceilMode: 0,
  countIncludePad: false,
  kernelShape: [],
  strides: [],
  pads: [],
  storageOrder: 0,
  dilations: [],
  cacheKey: ''
};

export const parseGlobalAveragePoolAttributes = (attributes: Record<string, unknown>): AveragePoolAttributes => {
  const format = attributes.format as FormatAttributes['format'];
  return {format, ...globalPoolAttributes, cacheKey: format};
};

export const globalAveragePool = (context: ComputeContext, attributes: AveragePoolAttributes): void => {
  validateInputs(context.inputs);
  context.compute(createAveragePoolProgramInfo('GlobalAveragePool', context.inputs[0], true, attributes));
};

export interface MaxPoolAttributes extends PoolCommonAttributes, AttributeWithCacheKey {
  readonly storageOrder: number;
  readonly dilations: number[];
}

const createMaxPoolProgramInfo =
    (name: string, input: TensorView, isGlobalOperator: boolean, attributes: MaxPoolAttributes): ProgramInfo => {
      const [adjustedAttributes, outputShape] =
          getAdjustedPoolAttributesAndOutputShape(input, attributes, isGlobalOperator);
      const op1 = `
      value = max(x_val, value);
    `;
      const op2 = '';
      const x = inputVariable('x', input.dataType, input.dims);
      return {
        name,
        shaderCache: {hint: attributes.cacheKey},
        getRunData: () => ({
          outputs: [{dims: outputShape, dataType: input.dataType}],
          dispatchGroup: {x: Math.ceil(ShapeUtil.size(outputShape) / 64 /* workgroup size */)}
        }),
        getShaderSource: shaderHelper =>
            generatePoolingCode(shaderHelper, x, input.dims, outputShape, adjustedAttributes, op1, op2, '-1e5'),
      };
    };

export const maxPool = (context: ComputeContext, attributes: MaxPoolAttributes): void => {
  validateInputs(context.inputs);
  context.compute(createMaxPoolProgramInfo('MaxPool', context.inputs[0], false, attributes));
};

export const parseMaxPoolAttributes = (attributes: Record<string, unknown>): MaxPoolAttributes => {
  const storageOrder = attributes.storage_order as number;
  const dilations = attributes.dilations as [number, number];

  const attr = parsePoolCommonAttributes(attributes);
  // TODO: support attribute 'ceil_mode' and 'storage_order'
  if (storageOrder !== 0) {
    throw new Error('column major storage order is not yet supported for MaxPool');
  }
  if (attr.ceilMode !== 0) {
    throw new Error('using ceil() in shape computation is not yet supported for MaxPool');
  }

  return createAttributeWithCacheKey({storageOrder, dilations, ...attr});
};

export const parseGlobalMaxPoolAttributes = (attributes: Record<string, unknown>): MaxPoolAttributes => {
  const format = attributes.format as FormatAttributes['format'];
  return {format, ...globalPoolAttributes, cacheKey: format};
};

export const globalMaxPool = (context: ComputeContext, attributes: MaxPoolAttributes): void => {
  validateInputs(context.inputs);
  context.compute(createMaxPoolProgramInfo('GlobalMaxPool', context.inputs[0], true, attributes));
};
