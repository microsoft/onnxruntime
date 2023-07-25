// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {DataType} from '../../../wasm-common';
import {TensorView} from '../../tensor';
import {PoolConvUtil, ShapeUtil} from '../../util';
import {AttributeWithCacheKey, createAttributeWithCacheKey} from '../attribute-with-cache-key';
import {ComputeContext, GpuDataType, ProgramInfo, ProgramMetadata} from '../types';

import {createIndicesHelper, ShaderHelper} from './common';

// TODO: support:
// - ceil_mode                 "test_maxpool_2d_ceil"
// - storage_order             "test_maxpool_with_argmax_2d_precomputed_strides"
// - [MaxPool] dilations       "test_maxpool_2d_dilations"
// - [MaxPool] output[1]       "test_maxpool_with_argmax_2d_precomputed_pads"

const validateInputs = (inputs: readonly TensorView[]): void => {
  if (!inputs || inputs.length !== 1) {
    throw new Error('Pool ops requires 1 input.');
  }
  if (inputs[0].dims.length !== 4) {
    throw new Error('Pool ops supports 2-D inputs only for now.');
  }
  if (inputs[0].dataType !== DataType.float) {
    throw new Error('Invalid input type.');
  }
};

const getAdjustedPoolAttributesAndOutputShape = <AttributeType extends AveragePoolAttributes|MaxPoolAttributes>(
    inputs: readonly TensorView[], attributes: AttributeType, isGlobalOperator: boolean): [AttributeType, number[]] => {
  const isChannelsLast = attributes.format === 'NHWC';
  const inputShapeAsChannelFirst = isChannelsLast ?
      [inputs[0].dims[0], inputs[0].dims[3], inputs[0].dims[1], inputs[0].dims[2]] :
      inputs[0].dims.slice();
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
  return [
    newAttributes,
    isChannelsLast ?
        [
          outputShapeAsChannelFirst[0], outputShapeAsChannelFirst[2], outputShapeAsChannelFirst[3],
          outputShapeAsChannelFirst[1]
        ] :
        outputShapeAsChannelFirst
  ];
};

const generatePoolingCode = <AttributeType extends AveragePoolAttributes|MaxPoolAttributes>(
    shaderHelper: ShaderHelper, inputDims: readonly number[], outputShape: readonly number[], attributes: AttributeType,
    op1: string, op2: string, dataType: string, start: string): string => {
  const isChannelsLast = attributes.format === 'NHWC';
  const rank = inputDims.length;
  const outputSize = ShapeUtil.size(outputShape);
  const outputIndicesHelper = createIndicesHelper('output', outputShape);
  const xIndicesHelper = createIndicesHelper('x', inputDims);

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
                let x_val = x[${xIndicesHelper.i2oExpression('xIndices')}];
                ${op1}
              }`;
    } else {
      codeW = `
              for (var i: u32 = 0u; i < ${kw}u; i++) {
                xIndices[${dimIdxW}] = indices[${dimIdxW}] * ${sw} - ${pwStart} + i;
                let x_val = x[${xIndicesHelper.i2oExpression('xIndices')}];
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
            @group(0) @binding(0) var<storage, read> x : array<${dataType}>;
            @group(0) @binding(1) var<storage, read_write> output : array<${dataType}>;

            ${outputIndicesHelper.o2iImpl}
            ${xIndicesHelper.i2oImpl}

            ${shaderHelper.mainStart()}
              ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes(outputSize)}

              ${outputIndicesHelper.indicesVariableDeclaration('indices')}
              ${outputIndicesHelper.o2iCall('global_idx', 'indices')}
              ${outputIndicesHelper.indicesVariableDeclaration('xIndices')}
              ${outputIndicesHelper.o2iCall('global_idx', 'xIndices')}

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
                let x_val = x[${xIndicesHelper.i2oExpression('xIndices')}];
                ${op1}
              }`;
    } else {
      padCode = `
              }
              let x_val = x[${xIndicesHelper.i2oExpression('xIndices')}];
              ${op1}
            `;
    }
    const poolingCode = `
            @group(0) @binding(0) var<storage, read> x : array<${dataType}>;
            @group(0) @binding(1) var<storage, read_write> output : array<${dataType}>;

            ${outputIndicesHelper.o2iImpl}
            ${xIndicesHelper.i2oImpl}

            const pads = array<u32, ${padsRank}>(${attributes.pads.map(i => `${i}u`).join(',')});
            const inputDims = array<u32, ${rank}>(${inputDims.map(i => `${i}u`).join(',')});
            const kernelStrides = array<u32, ${stridesRank}>(${kernelStrides.map(i => `${i}u`).join(',')});
            const strides = array<u32, ${stridesRank}>(${attributes.strides.map(i => `${i}u`).join(',')});

            ${shaderHelper.mainStart()}
              ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes(outputSize)}

              ${outputIndicesHelper.indicesVariableDeclaration('indices')}
              ${outputIndicesHelper.o2iCall('global_idx', 'indices')}
              ${outputIndicesHelper.indicesVariableDeclaration('xIndices')}
              ${outputIndicesHelper.o2iCall('global_idx', 'xIndices')}

              var offsets: array<u32, ${stridesRank}>;

              var value = ${dataType}(${start});
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
    (inputs: readonly TensorView[], metadata: ProgramMetadata, isGlobalOperator: boolean,
     attributes: AveragePoolAttributes): ProgramInfo => {
      const [adjustedAttributes, outputShape] =
          getAdjustedPoolAttributesAndOutputShape(inputs, attributes, isGlobalOperator);
      const kernelSize = ShapeUtil.size(adjustedAttributes.kernelShape);

      const dataType = 'f32';

      const op1 = 'value += x_val;';
      let op2 = '';
      if (adjustedAttributes.countIncludePad) {
        op2 += `value /= ${dataType}(${kernelSize});`;
      } else {
        op2 += `value /= ${dataType}(${kernelSize} - pad);`;
      }
      return {
        ...metadata,
        outputs: [{dims: outputShape, dataType: inputs[0].dataType, gpuDataType: GpuDataType.default}],
        getShaderSource: shaderHelper => generatePoolingCode(
            shaderHelper, inputs[0].dims, outputShape, adjustedAttributes, op1, op2, dataType, '0.0'),
        dispatchGroup: () => ({x: Math.ceil(ShapeUtil.size(outputShape) / 64 /* workgroup size */)})
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
  const metadata = {name: 'AveragePool', inputTypes: [GpuDataType.default], cacheHint: attributes.cacheKey};
  context.compute({...metadata, get: () => createAveragePoolProgramInfo(context.inputs, metadata, false, attributes)});
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
  const metadata = {name: 'GlobalAveragePool', inputTypes: [GpuDataType.default], cacheHint: attributes.cacheKey};
  context.compute({...metadata, get: () => createAveragePoolProgramInfo(context.inputs, metadata, true, attributes)});
};

export interface MaxPoolAttributes extends PoolCommonAttributes, AttributeWithCacheKey {
  readonly storageOrder: number;
  readonly dilations: number[];
}

const createMaxPoolProgramInfo =
    (inputs: readonly TensorView[], metadata: ProgramMetadata, isGlobalOperator: boolean,
     attributes: MaxPoolAttributes): ProgramInfo => {
      const [adjustedAttributes, outputShape] =
          getAdjustedPoolAttributesAndOutputShape(inputs, attributes, isGlobalOperator);
      const op1 = `
      value = max(x_val, value);
    `;
      const op2 = '';
      return {
        ...metadata,
        outputs: [{dims: outputShape, dataType: inputs[0].dataType, gpuDataType: GpuDataType.default}],
        getShaderSource: shaderHelper =>
            generatePoolingCode(shaderHelper, inputs[0].dims, outputShape, adjustedAttributes, op1, op2, 'f32', '-1e5'),
        dispatchGroup: () => ({x: Math.ceil(ShapeUtil.size(outputShape) / 64 /* workgroup size */)})
      };
    };

export const maxPool = (context: ComputeContext, attributes: MaxPoolAttributes): void => {
  validateInputs(context.inputs);
  const metadata = {name: 'MaxPool', inputTypes: [GpuDataType.default], cacheHint: attributes.cacheKey};
  context.compute({...metadata, get: () => createMaxPoolProgramInfo(context.inputs, metadata, false, attributes)});
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
  const metadata = {name: 'GlobalMaxPool', inputTypes: [GpuDataType.default], cacheHint: attributes.cacheKey};
  context.compute({...metadata, get: () => createMaxPoolProgramInfo(context.inputs, metadata, true, attributes)});
};
