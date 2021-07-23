// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {Graph} from '../../../graph';
import {OperatorImplementation, OperatorInitialization} from '../../../operators';
import {Tensor} from '../../../tensor';
import {PoolConvUtil, ShapeUtil} from '../../../util';
import {WebGLInferenceHandler} from '../inference-handler';
import {ProgramInfo, TextureType} from '../types';

export interface AveragePoolAttributes {
  autoPad: string;
  ceilMode: number;
  countIncludePad: boolean;
  kernelShape: number[];
  strides: number[];
  pads: number[];
}

export const averagePool: OperatorImplementation<AveragePoolAttributes> =
    (inferenceHandler: WebGLInferenceHandler, inputs: Tensor[], attributes: AveragePoolAttributes): Tensor[] => {
      validateInputs(inputs);
      const output =
          inferenceHandler.run(createAveragePoolProgramInfo(inferenceHandler, inputs, false, attributes), inputs);
      return [output];
    };

export const parseAveragePoolAttributes: OperatorInitialization<AveragePoolAttributes> =
    (node: Graph.Node): AveragePoolAttributes => {
      const autoPad = node.attributes.getString('auto_pad', 'NOTSET');
      const ceilMode = node.attributes.getInt('ceil_mode', 0);
      const countIncludePad = (node.attributes.getInt('count_include_pad', 0) === 0 ? false : true);
      const kernelShape = node.attributes.getInts('kernel_shape');
      const strides = node.attributes.getInts('strides', []);
      const pads = node.attributes.getInts('pads', []);

      // TODO: support attribute 'ceil_mode'
      if (ceilMode !== 0) {
        throw new Error('using ceil() in shape computation is not yet supported for AveragePool');
      }

      return {autoPad, ceilMode, countIncludePad, kernelShape, strides, pads};
    };

const createAveragePoolProgramInfo =
    (inferenceHandler: WebGLInferenceHandler, inputs: Tensor[], isGlobalOperator: boolean,
     attributes: AveragePoolAttributes): ProgramInfo => {
      const inputShape = inputs[0].dims.slice();
      PoolConvUtil.adjustPoolAttributes(
          isGlobalOperator, inputShape, attributes.kernelShape, attributes.strides, attributes.pads);
      const outputShape = PoolConvUtil.computePoolOutputShape(
          isGlobalOperator, inputShape, attributes.strides, attributes.kernelShape, attributes.pads,
          attributes.autoPad);
      const kernelSize = ShapeUtil.size(attributes.kernelShape);
      const op1 = 'value += _X(x);';
      let op2 = '';
      if (attributes.countIncludePad) {
        op2 += `value /= float(${kernelSize});`;
      } else {
        op2 += `value /= float(${kernelSize} - pad);`;
      }
      const poolingCode = generatePoolingCode(inputs[0].dims, attributes, op1, op2, '0.0');
      const shaderSource = `
        ${poolingCode}
      `;
      return {
        name: (isGlobalOperator) ? 'GlobalAveragePool' : 'AveragePool',
        inputNames: ['X'],
        inputTypes: [TextureType.unpacked],
        output: {dims: outputShape, type: inputs[0].type, textureType: TextureType.unpacked},
        shaderSource
      };
    };

export const globalAveragePool: OperatorImplementation<AveragePoolAttributes> =
    (inferenceHandler: WebGLInferenceHandler, inputs: Tensor[], attributes: AveragePoolAttributes): Tensor[] => {
      validateInputs(inputs);
      const output =
          inferenceHandler.run(createAveragePoolProgramInfo(inferenceHandler, inputs, true, attributes), inputs);
      return [output];
    };

export const parseGlobalAveragePoolAttributes: OperatorInitialization<AveragePoolAttributes> =
    (node: Graph.Node): AveragePoolAttributes => {
      const countIncludePad = (node.attributes.getInt('count_include_pad', 0) === 0 ? false : true);
      return {autoPad: '', ceilMode: 0, countIncludePad, kernelShape: [], strides: [], pads: []};
    };

export interface MaxPoolAttributes extends AveragePoolAttributes {
  storageOrder: number;
}

export const maxPool: OperatorImplementation<MaxPoolAttributes> =
    (inferenceHandler: WebGLInferenceHandler, inputs: Tensor[], attributes: MaxPoolAttributes): Tensor[] => {
      validateInputs(inputs);
      const output =
          inferenceHandler.run(createMaxPoolProgramInfo(inferenceHandler, inputs, false, attributes), inputs);
      return [output];
    };

export const parseMaxPoolAttributes: OperatorInitialization<MaxPoolAttributes> =
    (node: Graph.Node): MaxPoolAttributes => {
      const autoPad = node.attributes.getString('auto_pad', 'NOTSET');
      const ceilMode = node.attributes.getInt('ceil_mode', 0);
      const kernelShape = node.attributes.getInts('kernel_shape');
      const strides = node.attributes.getInts('strides', []);
      const pads = node.attributes.getInts('pads', []);
      const storageOrder = node.attributes.getInt('storage_order', 0);

      // TODO: support attribute 'ceil_mode' and 'storage_order'
      if (storageOrder !== 0) {
        throw new Error('column major storage order is not yet supported for MaxPool');
      }
      if (ceilMode !== 0) {
        throw new Error('using ceil() in shape computation is not yet supported for MaxPool');
      }

      return {autoPad, ceilMode, countIncludePad: false, kernelShape, strides, pads, storageOrder};
    };

const createMaxPoolProgramInfo =
    (inferenceHandler: WebGLInferenceHandler, inputs: Tensor[], isGlobalOperator: boolean,
     attributes: MaxPoolAttributes): ProgramInfo => {
      const inputShape = inputs[0].dims.slice();
      PoolConvUtil.adjustPoolAttributes(
          isGlobalOperator, inputShape, attributes.kernelShape, attributes.strides, attributes.pads);
      const outputShape = PoolConvUtil.computePoolOutputShape(
          isGlobalOperator, inputShape, attributes.strides, attributes.kernelShape, attributes.pads,
          attributes.autoPad);
      const op1 = `
      value = max(_X(x), value);
    `;
      const op2 = '';
      const poolingCode = generatePoolingCode(inputShape, attributes, op1, op2, '-1e5');
      const shaderSource = `
      ${poolingCode}
    `;
      return {
        name: (isGlobalOperator) ? 'GlobalMaxPool' : 'MaxPool',
        inputNames: ['X'],
        inputTypes: [TextureType.unpacked],
        output: {dims: outputShape, type: inputs[0].type, textureType: TextureType.unpacked},
        shaderSource
      };
    };

export const globalMaxPool = (inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): Tensor[] => {
  validateInputs(inputs);
  const attributes: MaxPoolAttributes =
      {autoPad: '', ceilMode: 0, countIncludePad: false, kernelShape: [], strides: [], pads: [], storageOrder: 0};
  const output = inferenceHandler.run(createMaxPoolProgramInfo(inferenceHandler, inputs, true, attributes), inputs);
  return [output];
};

const validateInputs = (inputs: Tensor[]): void => {
  if (!inputs || inputs.length !== 1) {
    throw new Error('Pool ops requires 1 input.');
  }
  if (inputs[0].type !== 'float32' && inputs[0].type !== 'float64') {
    throw new Error('Invalid input type.');
  }
};

const generatePoolingCode =
    (inputDims: readonly number[], attributes: AveragePoolAttributes, op1: string, op2: string, start: string):
        string => {
          const rank = inputDims.length;
          if (attributes.kernelShape.length <= 2) {
            const kw = attributes.kernelShape[attributes.kernelShape.length - 1];
            const sw = attributes.strides[attributes.strides.length - 1];
            const pwStart = attributes.pads[attributes.pads.length / 2 - 1];
            const pwEnd = attributes.pads[attributes.pads.length - 1];
            const dimW = inputDims[rank - 1];
            let codeW = '';
            let codeH = '';
            let codeHEnd = '';
            if (pwStart + pwEnd !== 0) {
              codeW = `
          for (int i = 0; i < ${kw}; i++) {
            x[${rank} - 1] = indices[${rank} - 1] * ${sw} - ${pwStart} + i;
            if (x[${rank} - 1] < 0 || x[${rank} - 1] >= ${dimW}) {
              pad++;
              continue;
            }
            ${op1}
          }`;
            } else {
              codeW = `
          for (int i = 0; i < ${kw}; i++) {
            x[${rank} - 1] = indices[${rank} - 1] * ${sw} - ${pwStart} + i;
            ${op1}
          }`;
            }

            if (attributes.kernelShape.length === 2) {
              const kh = attributes.kernelShape[attributes.kernelShape.length - 2];
              const sh = attributes.strides[attributes.strides.length - 2];
              const phStart = attributes.pads[attributes.pads.length / 2 - 2];
              const phEnd = attributes.pads[attributes.pads.length - 2];
              const dimH = inputDims[rank - 2];
              if (phStart + phEnd !== 0) {
                codeH = `
            for (int j = 0; j < ${kh}; j++) {
              x[${rank} - 2] = indices[${rank} - 2] * ${sh} - ${phStart} + j;
              if (x[${rank} - 2] < 0 || x[${rank} - 2] >= ${dimH}) {
                pad+= ${kw};
                continue;
              }
          `;
              } else {
                codeH = `
            for (int j = 0; j < ${kh}; j++) {
              x[${rank} - 2] = indices[${rank} - 2] * ${sh} - ${phStart} + j;
            `;
              }
              codeHEnd = `
          }
        `;
            }

            const poolingCode = `
        float process(int indices[${rank}]) {
          int x[${rank}];
          copyVec(indices, x);

          float value = ${start};
          int pad = 0;
          ${codeH}
          ${codeW}
          ${codeHEnd}
          ${op2}
          return value;
        }
      `;
            return poolingCode;
          } else {
            const kernelSize = ShapeUtil.size(attributes.kernelShape);
            const kernelStrides = ShapeUtil.computeStrides(attributes.kernelShape);
            const stridesRank = kernelStrides.length;
            const padsRank = attributes.pads.length;
            const offsetToIndicesFunction = offsetToIndices(stridesRank);
            const copyInputDims = copyArray(inputDims, 'inputDims');
            const copyPads = copyArray(attributes.pads, 'pads');
            const copyKernelStrides = copyArray(kernelStrides, 'kernelStrides');
            const copyStrides = copyArray(attributes.strides, 'strides');
            const hasPads = attributes.pads.reduce((sum, cur) => sum + cur);
            let padCode = '';
            if (hasPads) {
              padCode = `
            if (x[j] >= inputDims[j] || x[j] < 0) {
              pad++;
              isPad = true;
              break;
            }
          }
          if (!isPad) {
            ${op1}
          }`;
            } else {
              padCode = `
          }
          ${op1}
        `;
            }
            const poolingCode = `
        ${offsetToIndicesFunction}
        float process(int indices[${rank}]) {
          int x[${rank}];
          copyVec(indices, x);
          int offset[${stridesRank}];
          int pads[${padsRank}];
          int inputDims[${rank}];
          int kernelStrides[${stridesRank}];
          int strides[${stridesRank}];
          ${copyPads}
          ${copyInputDims}
          ${copyStrides}
          ${copyKernelStrides}

          float value = ${start};
          int pad = 0;
          bool isPad = false;
          for (int i = 0; i < ${kernelSize}; i++) {
            offsetToIndices(i, kernelStrides, offset);
            isPad = false;
            for (int j = ${rank} - ${stridesRank}; j < ${rank}; j++) {
              x[j] = indices[j] * strides[j - ${rank} + ${stridesRank}]
                + offset[j - ${rank} + ${stridesRank}] - pads[j - 2];
              ${padCode}
          }
          ${op2}

          return value;
        }
      `;
            return poolingCode;
          }
        };

const copyArray = (array: readonly number[], arrayName: string): string => {
  let block = '';
  for (let i = 0; i < array.length; i++) {
    block += `
      ${arrayName}[${i}] = ${array[i]};
    `;
  }
  return block;
};

const offsetToIndices = (rank: number): string => `
  void offsetToIndices(int offset, int[${rank}] strides, out int[${rank}] indices) {
    if (${rank} == 0) {
      return;
    }
    for (int i = 0; i < ${rank} - 1; ++i) {
      indices[i] = offset / strides[i];
      offset -= indices[i] * strides[i];
    }
    indices[${rank} - 1] = offset;
  }`;