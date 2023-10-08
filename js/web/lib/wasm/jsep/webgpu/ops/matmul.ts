// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {TensorView} from '../../tensor-view';

import {createMatmulProgramInfo} from './3rd-party/matmul_packed_webgpu';
import {BroadcastUtil, ShapeUtil} from '../../util';
import {ComputeContext, ProgramInfo} from '../types';

import {getBroadcastDims, IndicesHelper, inputVariable, outputVariable, ShaderHelper,} from './common';
import {getActivationSnippet, InternalActivationAttributes} from './fuse-utils';

export const createNaiveMatmulProgramInfo =
    (inputs: readonly TensorView[], activationAttributes: InternalActivationAttributes,
     outputShape: readonly number[], reshapedOutputShape?: readonly number[],
     isChannelsLast = false /* only used for conv2dByMatMul*/): ProgramInfo => {
      const aShape = inputs[0].dims;
      const bShape = inputs[1].dims;
      const outputSize = ShapeUtil.size(outputShape);

      const M = aShape[aShape.length - 2];
      const K = aShape[aShape.length - 1];
      const N = bShape[bShape.length - 1];
      const outerDims = reshapedOutputShape ? reshapedOutputShape.slice(0, -2) : outputShape.slice(0, -2);
      const batchSize = ShapeUtil.size(outerDims);
      const components = 1;
      const a = inputVariable('a', inputs[0].dataType, aShape, components);
      const b = inputVariable('b', inputs[1].dataType, bShape, components);
      const output = outputVariable('output', inputs[0].dataType, [batchSize, M, N], components);
      const {activationFunction, applyActivation} = getActivationSnippet(activationAttributes, output.type.value);
      const inputVariables = [a, b];
      const hasBias = inputs.length > 2;
      if (hasBias) {
        const biasComponents = isChannelsLast ? components : 1;
        inputVariables.push(inputVariable('bias', inputs[2].dataType, inputs[2].dims, biasComponents));
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

      const getShaderSource = (shaderHelper: ShaderHelper) => `
  const M: u32 = ${M}u;
  const N: u32 = ${N}u;
  const K: u32 = ${K}u;
  ${shaderHelper.declareVariables(...inputVariables, output)}
  ${activationFunction}
  ${shaderHelper.mainStart()}
    ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes(outputSize)}
    let outputIndices = ${output.offsetToIndices('global_idx')};
    let batch = outputIndices.x;
    let row = outputIndices.y;
    let col = outputIndices.z;
    ${outputShape.length === 2 ? '' : `let batchIndices = ${batchDims.offsetToIndices('batch')};`}
    ${getIndices(a, broadCastADims)}
    let offsetA = ${a.indicesToOffset('aIndices')};
    ${getIndices(b, broadCastBDims)}
    let offsetB = ${b.indicesToOffset('bIndices')};
    var value = ${output.type.value}(0);
    for (var k: u32 = 0u; k<${K}u; k++) {
      value += a[offsetA + row * K + k] * b[offsetB + k * N + col];
    }
    ${applyActivation}
    output[global_idx] = value;
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
