// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {DataType} from '../../../wasm-common';
import {TensorView} from '../../tensor-view';
import {ShapeUtil} from '../../util';
import {AttributeWithCacheKey, createAttributeWithCacheKey} from '../attribute-with-cache-key';
import {ComputeContext, ProgramInfo} from '../types';

import {IndicesHelper, inputVariable, outputVariable, ShaderHelper} from './common';

export interface PadAttributes extends AttributeWithCacheKey {
  // 0-constant, 1-reflect, 2-edge, 3-wrap
  readonly mode: number;
  readonly value: number;
  readonly pads: number[];
}

const validateInputs = (inputs: readonly TensorView[]): void => {
  if (!inputs || inputs.length < 1) {
    throw new Error('Too few inputs');
  }
  if (inputs[0].dataType !== DataType.float) {
    throw new Error('Input type must be float.');
  }

  if (inputs.length >= 2) {
    let validPads = inputs[0].dims.length * 2 === inputs[1].dims[0];
    if (inputs.length === 4) {
      validPads = inputs[3].dims[0] * 2 === inputs[1].dims[0];
    }
    if (!validPads) {
      throw new Error('The pads should be a 1D tensor of shape [2 * input_rank] or [2 * num_axes].');
    }
  }
};

const getPadConstant =
    (output: IndicesHelper, outputDims: readonly number[], inputDims: readonly number[],
     inputStrides: readonly number[], pads: number[], dataType: string, constantValue: number): string => {
      const inputRank = inputDims.length;

      let block = '';
      for (let i = inputRank - 1; i >= 0; --i) {
        block += `
            k = i32(${output.indicesGet('indices', i)}) - ${pads[i]};
            if (k < 0) {
              break;
            }
            if (k >= ${inputDims[i]}) {
              break;
            }
            offset += k * ${inputStrides[i]};
        `;
      }

      return `
          value = ${dataType}(${constantValue});
          for (var i = 0; i < 1; i++) {
            var offset = 0;
            var k = 0;
            ${block}
            value = x[offset];
          }
      `;
    };

const getPadReflect =
    (output: IndicesHelper, outputDims: readonly number[], inputDims: readonly number[],
     inputStrides: readonly number[], pads: number[]): string => {
      const inputRank = inputDims.length;

      let block = '';
      for (let i = inputRank - 1; i >= 0; --i) {
        block += `
                k = i32(${output.indicesGet('indices', i)}) - ${pads[i]};
                if (k < 0) {
                  k = -k;
                }
                {
                  let _2n_1 = ${2 * (inputDims[i] - 1)};
                  k = k % _2n_1;
                  if(k >= ${inputDims[i]}) {
                    k = _2n_1 - k;
                  }
                }
                offset += k * ${inputStrides[i]};
            `;
      }

      return `
              var offset = 0;
              var k = 0;
              ${block}
              value = x[offset];
          `;
    };

const getPadEdge =
    (output: IndicesHelper, outputDims: readonly number[], inputDims: readonly number[],
     inputStrides: readonly number[], pads: number[]): string => {
      const inputRank = inputDims.length;

      let block = '';
      for (let i = inputRank - 1; i >= 0; --i) {
        block += `
                k = i32(${output.indicesGet('indices', i)}) - ${pads[i]};
                if (k < 0) {
                  k = 0;
                }
                if (k >= ${inputDims[i]}) {
                  k = ${inputDims[i] - 1};
                }
                offset += k * ${inputStrides[i]};
            `;
      }

      return `
              var offset = 0;
              var k = 0;
              ${block}
              value = x[offset];
          `;
    };

const getPadWrap =
    (output: IndicesHelper, outputDims: readonly number[], inputDims: readonly number[],
     inputStrides: readonly number[], pads: number[]): string => {
      const inputRank = inputDims.length;

      let block = '';
      for (let i = inputRank - 1; i >= 0; --i) {
        block += `
                k = i32(${output.indicesGet('indices', i)}) - ${pads[i]};
                if (k < 0)  {
                  k += ${inputDims[i]};
                }
                if (k >= ${inputDims[i]}) {
                  k -= ${inputDims[i]};
                }
                offset += k * ${inputStrides[i]};
            `;
      }

      return `
              var offset = 0;
              var k = 0;
              ${block}
              value = x[offset];
          `;
    };

const getPadSnippet =
    (output: IndicesHelper, outputDims: readonly number[], inputDims: readonly number[],
     inputStrides: readonly number[], attributes: PadAttributes, dataType: string): string => {
      switch (attributes.mode) {
        case 0:
          return getPadConstant(
              output, outputDims, inputDims, inputStrides, attributes.pads, dataType, attributes.value);
        case 1:
          return getPadReflect(output, outputDims, inputDims, inputStrides, attributes.pads);
        case 2:
          return getPadEdge(output, outputDims, inputDims, inputStrides, attributes.pads);
        case 3:
          return getPadWrap(output, outputDims, inputDims, inputStrides, attributes.pads);
        default:
          throw new Error('Invalid mode');
      }
    };

const generatePadCode =
    (shaderHelper: ShaderHelper, inputs: readonly TensorView[], attributes: PadAttributes, dataType: string):
        string => {
          const inputDims = inputs[0].dims;
          const outputDims = ShapeUtil.padShape(inputDims.slice(), attributes.pads);
          const outputSize = ShapeUtil.size(outputDims);
          const inputStrides = ShapeUtil.computeStrides(inputDims);

          const output = outputVariable('output', inputs[0].dataType, outputDims);
          const input = inputVariable('x', inputs[0].dataType, inputDims);

          const padSnippet = getPadSnippet(output, outputDims, inputDims, inputStrides, attributes, dataType);
          const padCode = `
              ${shaderHelper.declareVariables(input, output)}
              ${shaderHelper.mainStart()}
              ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes(outputSize)}

              let indices = ${output.offsetToIndices('global_idx')};

              var value = ${dataType}(0);
              ${padSnippet}
              output[global_idx] = value;
          }`;
          return padCode;
        };

const createPadProgramInfo = (inputs: readonly TensorView[], attributes: PadAttributes): ProgramInfo => {
  const outputShape = ShapeUtil.padShape(inputs[0].dims.slice(), attributes.pads);
  return {
    name: 'Pad',
    shaderCache: {hint: attributes.cacheKey},
    getRunData: () => ({
      outputs: [{dims: outputShape, dataType: inputs[0].dataType}],
      dispatchGroup: {x: Math.ceil(ShapeUtil.size(outputShape) / 64 /* workgroup size */)}
    }),
    getShaderSource: shaderHelper => generatePadCode(shaderHelper, inputs, attributes, 'f32'),
  };
};

const createPadAttributesFromInputs = (inputs: readonly TensorView[], attributes: PadAttributes): PadAttributes => {
  if (inputs.length > 1) {
    const bigInt64Pads = inputs[1].getBigInt64Array();
    const value = (inputs.length >= 3 && inputs[2].data) ? inputs[2].getFloat32Array()[0] : 0.0;

    const inputRank = inputs[0].dims.length;
    const updatePads = new Int32Array(2 * inputRank).fill(0);
    if (inputs.length >= 4) {
      const axes = inputs[3].getBigInt64Array();
      for (let i = 0; i < axes.length; i++) {
        updatePads[Number(axes[i])] = Number(bigInt64Pads[i]);
        updatePads[Number(axes[i]) + inputRank] = Number(bigInt64Pads[i + axes.length]);
      }
    } else {
      bigInt64Pads.forEach((v, i) => updatePads[Number(i)] = (Number(v)));
    }

    const pads: number[] = [];
    updatePads.forEach(v => pads.push(v));

    return createAttributeWithCacheKey({mode: attributes.mode, value, pads});
  } else {
    return attributes;
  }
};

export const pad = (context: ComputeContext, attributes: PadAttributes): void => {
  validateInputs(context.inputs);
  const updatedAttributes = createPadAttributesFromInputs(context.inputs, attributes);
  context.compute(createPadProgramInfo(context.inputs, updatedAttributes), {inputs: [0]});
};

export const parsePadAttributes = (attributes: Record<string, unknown>): PadAttributes => {
  const mode = attributes.mode as number;
  const value = attributes.value as number;
  const pads = attributes.pads as number[];
  return createAttributeWithCacheKey({mode, value, pads});
};
