// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {TensorView} from '../../tensor-view';

import {createMatmulProgramInfo} from './3rd-party/matmul_packed_webgpu';
import {BroadcastUtil, ShapeUtil} from '../../util';
import {ComputeContext, ProgramInfo} from '../types';

import {getBroadcastDims, IndicesHelper, inputVariable, outputVariable, ShaderHelper,} from './common';
import {getActivationSnippet, InternalActivationAttributes} from './fuse-utils';

const getMaxComponents = (size: number): 1|2|3|4 => {
  if (size % 4 === 0) {
    return 4;
  } else if (size % 2 === 0) {
    return 2;
  }
  return 1;
};
export const createNaiveMatmulProgramInfo =
    (inputs: readonly TensorView[], activationAttributes: InternalActivationAttributes,
     outputShape: readonly number[], reshapedOutputShape?: readonly number[],
     isChannelsLast = false /* only used for conv2dByMatMul*/): ProgramInfo => {
      const aShape = inputs[0].dims;
      const bShape = inputs[1].dims;

      const M = aShape[aShape.length - 2];
      const N = bShape[bShape.length - 1];
      const K = aShape[aShape.length - 1];
      const outerDims = reshapedOutputShape ? reshapedOutputShape.slice(0, -2) : outputShape.slice(0, -2);
      const batchSize = ShapeUtil.size(outerDims);
      const components = getMaxComponents(N);
      const aComponents = getMaxComponents(K);
      const outputNumber = getMaxComponents(M);
      const outputSize = ShapeUtil.size(outputShape) / components / outputNumber;
      const a = inputVariable('a', inputs[0].dataType, aShape, aComponents);
      const b = inputVariable('b', inputs[1].dataType, bShape, components);
      const output = outputVariable('output', inputs[0].dataType, [batchSize, M, N], components);
      const {activationFunction, applyActivation} = getActivationSnippet(activationAttributes, output.type.value);
      const inputVariables = [a, b];
      const hasBias = inputs.length > 2;
      let processBias = '';
      if (hasBias) {
        const biasComponents = isChannelsLast ? components : 1;
        inputVariables.push(inputVariable('bias', inputs[2].dataType, inputs[2].dims, biasComponents));
        processBias = `${
            isChannelsLast ? `value += bias[col / ${biasComponents}];` :
                             `value += ${output.type.value}(bias[row + i]);`}`;
      }
      const outerDimsA = aShape.slice(0, -2);
      const outerDimsB = bShape.slice(0, -2);
      const batchDims = inputVariable('batchDims', inputs[0].dataType, outerDims);
      const broadCastADims = getBroadcastDims(outerDimsA, outerDims);
      const broadCastBDims = getBroadcastDims(outerDimsB, outerDims);
      const getIndices = (variable: IndicesHelper, broadCastDims: number[]) => {
        const rank = variable.shape.length;
        const name = variable.name;
        if (rank === 2) {
          return `var ${name}Indices = ${variable.type.indices}(0u, 0u);`;
        }
        const batchRank = batchDims.shape.length;
        let resStr = `var ${name}Indices: ${variable.type.indices};`;
        for (let i = rank - 2 - 1, j = batchRank - 1; i >= 0; i--, j--) {
          resStr += `\n${name}Indices[${i}] = ${batchRank > 1 ? `batchIndices[${j}]` : 'batchIndices'};`;
        }
        broadCastDims.forEach(i => {
          resStr += `\n${name}Indices[${i}] = 0;`;
        });
        resStr += `${name}Indices[${rank - 2}] = 0u;
                   ${name}Indices[${rank - 1}] = 0u;`;
        return resStr;
      };

      const calcResult = (): string => {
        let calcStr = `var aData: ${a.type.value};`;
        for (let i = 0; i < aComponents; i++) {
          calcStr += `
            let bData${i} = b[(offsetB + (k + ${i}) * N + col) / ${components}];`;
        }
        for (let i = 0; i < outputNumber; i++) {
          calcStr += `aData = a[(offsetA + (row + ${i}) * K + k) / ${aComponents}];`;

          for (let j = 0; j < aComponents; j++) {
            calcStr += `
          values[${i}] = fma(${b.type.value}(aData${aComponents === 1 ? '' : `[${j}]`}), bData${j}, values[${i}]);\n`;
          }
        }
        return calcStr;
      };

      const getShaderSource = (shaderHelper: ShaderHelper) => `
  const M: u32 = ${M}u;
  const N: u32 = ${N}u;
  const K: u32 = ${K}u;
  ${shaderHelper.declareVariables(...inputVariables, output)}
  ${activationFunction}
  ${shaderHelper.mainStart()}
    ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes(outputSize)}
    let col = (global_idx % ${N / components}u) * ${components}u;
    var index1 = global_idx / ${N / components}u;
    let stride1 = ${M / outputNumber}u;
    let row = (index1 % stride1) * ${outputNumber}u;
    let batch = index1 / stride1;

    ${outputShape.length === 2 ? '' : `let batchIndices = ${batchDims.offsetToIndices('batch')};`}
    ${getIndices(a, broadCastADims)}
    let offsetA = ${a.indicesToOffset('aIndices')};
    ${getIndices(b, broadCastBDims)}
    let offsetB = ${b.indicesToOffset('bIndices')};
    var values: array<${output.type.value}, ${outputNumber}>;
    for (var k: u32 = 0u; k < K; k = k + ${aComponents}) {
      ${calcResult()}
    }
    for (var i = 0u; i < ${outputNumber}u; i++) {
      var value = values[i];
      ${processBias}
      ${applyActivation}
      let curIndices = ${output.type.indices}(batch, row + i, col);
      let offset = ${output.indicesToOffset('curIndices')};
      ${output.setByOffset(`offset / ${components}`, 'value')};
    }
  }
  ${batchDims.impl()}
  `;
      return {
        name: 'MatMulNaive',
        shaderCache: {hint: activationAttributes.activationCacheKey},
        getRunData: () => ({outputs: [{dims: outputShape, dataType: inputs[0].dataType}],
                            dispatchGroup: {x: Math.ceil(outputSize / 64 /* workgroup size */)}}),
        getShaderSource
      };
    };

const validateInputs = (inputs: readonly TensorView[]): void => {
  if (!inputs || inputs.length !== 2) {
    throw new Error('MatMul requires 2 inputs.');
  }

  if (inputs[0].dims[inputs[0].dims.length - 1] !== inputs[1].dims[inputs[1].dims.length - 2]) {
    throw new Error('shared dimension does not match.');
  }
};

export const matMul = (context: ComputeContext): void => {
  validateInputs(context.inputs);
  const outputShape = BroadcastUtil.calcShape(context.inputs[0].dims, context.inputs[1].dims, true);
  if (!outputShape) {
    throw new Error('Can\'t use matmul on the given tensors');
  }
  context.compute(createMatmulProgramInfo(context.inputs, {activation: '', activationCacheKey: ''}, outputShape));
};
