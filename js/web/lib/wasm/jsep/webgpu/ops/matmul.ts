// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {TensorView} from '../../tensor-view';
import {BroadcastUtil, ShapeUtil} from '../../util';
import {ComputeContext, ProgramInfo, ProgramUniform} from '../types';

import {createMatmulProgramInfo} from './3rd-party/matmul_packed_webgpu';
import {createTensorShapeVariables, getBroadcastDims, getMaxComponents, IndicesHelper, inputVariable, internalVariable, outputVariable, ShaderHelper, UniformsArrayType,} from './common';
import {getActivationSnippet, InternalActivationAttributes, updateUniformsFromActivation} from './fuse-utils';

export const createNaiveMatmulProgramInfo =
    (inputs: readonly TensorView[], activationAttributes: InternalActivationAttributes, outputShape: readonly number[],
     reshapedOutputShape?: readonly number[],
     isChannelsLast = false /* only used for conv2dByMatMul*/): ProgramInfo => {
      const aShape = inputs[0].dims;
      const bShape = inputs[1].dims;

      const M = aShape[aShape.length - 2];
      const N = bShape[bShape.length - 1];
      const K = aShape[aShape.length - 1];
      const components = getMaxComponents(N);
      const aComponents = getMaxComponents(K);
      const outputNumber = getMaxComponents(M);
      const outputSize = ShapeUtil.size(outputShape) / components / outputNumber;
      const hasBias = inputs.length > 2;
      const outerDims = reshapedOutputShape ? reshapedOutputShape.slice(0, -2) : outputShape.slice(0, -2);
      const batchSize = ShapeUtil.size(outerDims);
      const outputShapeInShader = [batchSize, M, N];

      const batchDims = internalVariable('batch_dims', inputs[0].dataType, outerDims.length);
      const a = inputVariable('a', inputs[0].dataType, aShape.length, aComponents);
      const b = inputVariable('b', inputs[1].dataType, bShape.length, components);
      const output = outputVariable('output', inputs[0].dataType, outputShapeInShader.length, components);
      const {activationFunction, applyActivation} = getActivationSnippet(activationAttributes, output.type.value);
      const inputVariables = [a, b];
      let processBias = '';
      if (hasBias) {
        const biasComponents = isChannelsLast ? components : 1;
        inputVariables.push(inputVariable('bias', inputs[2].dataType, inputs[2].dims.length, biasComponents));
        processBias = `${
            isChannelsLast ? `value += bias[col / ${biasComponents}];` :
                             `value += ${output.type.value}(bias[row + i]);`}`;
      }

      const outerDimsA = aShape.slice(0, -2);
      const outerDimsB = bShape.slice(0, -2);
      const broadCastADims = getBroadcastDims(outerDimsA, outerDims);
      const broadCastBDims = getBroadcastDims(outerDimsB, outerDims);

      const programUniforms: ProgramUniform[] = [
        {type: 'uint32', data: outputSize}, {type: 'uint32', data: M}, {type: 'uint32', data: N},
        {type: 'uint32', data: K}
      ];
      const uniforms: UniformsArrayType = [
        {name: 'outputSize', type: 'u32'}, {name: 'M', type: 'u32'}, {name: 'N', type: 'u32'}, {name: 'K', type: 'u32'}
      ];
      updateUniformsFromActivation(
          programUniforms, uniforms, activationAttributes, inputs[0].dataType, output.type.value);
      programUniforms.push(
          ...createTensorShapeVariables(outerDims), ...createTensorShapeVariables(aShape),
          ...createTensorShapeVariables(bShape));
      if (hasBias) {
        programUniforms.push(...createTensorShapeVariables(inputs[2].dims));
      }
      programUniforms.push(...createTensorShapeVariables(outputShapeInShader));

      const getShaderSource = (shaderHelper: ShaderHelper) => {
        const getIndices = (variable: IndicesHelper, broadCastDims: number[]) => {
          const rank = variable.rank;
          const name = variable.name;
          if (rank === 2) {
            return `var ${name}_indices = ${variable.type.indices}(0u, 0u);`;
          }
          const batchRank = batchDims.rank;
          let resStr = `var ${name}_indices: ${variable.type.indices};`;
          for (let i = rank - 2 - 1, j = batchRank - 1; i >= 0; i--, j--) {
            resStr += `\n${name}_indices[${i}] = ${batchRank > 1 ? `batch_indices[${j}]` : 'batch_indices'};`;
          }
          broadCastDims.forEach(i => {
            resStr += `\n${name}_indices[${i}] = 0;`;
          });
          resStr += `${name}_indices[${rank - 2}] = 0u;
                     ${name}_indices[${rank - 1}] = 0u;`;
          return resStr;
        };

        const calcResult = (): string => {
          let calcStr = `var a_data: ${a.type.value};`;
          for (let i = 0; i < aComponents; i++) {
            calcStr += `
              let b_data${i} = b[(b_offset + (k + ${i}) * uniforms.N + col) / ${components}];`;
          }
          for (let i = 0; i < outputNumber; i++) {
            calcStr += `a_data = a[(a_offset + (row + ${i}) * uniforms.K + k) / ${aComponents}];`;

            for (let j = 0; j < aComponents; j++) {
              calcStr += `
            values[${i}] = fma(${b.type.value}(a_data${aComponents === 1 ? '' : `[${j}]`}), b_data${j}, values[${
                  i}]);\n`;
            }
          }
          return calcStr;
        };

        return `
  ${
            shaderHelper.registerUniforms(uniforms).registerInternalVariables(batchDims).declareVariables(
                ...inputVariables, output)}
  ${activationFunction}
  ${shaderHelper.mainStart()}
    ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes('uniforms.outputSize')}
    let col = (global_idx % (uniforms.N / ${components})) * ${components};
    var index1 = global_idx / (uniforms.N / ${components});
    let stride1 = uniforms.M / ${outputNumber};
    let row = (index1 % stride1) * ${outputNumber};
    let batch = index1 / stride1;

    ${outputShape.length === 2 ? '' : `let batch_indices = ${batchDims.offsetToIndices('batch')};`}
    ${getIndices(a, broadCastADims)}
    let a_offset = ${a.indicesToOffset('a_indices')};
    ${getIndices(b, broadCastBDims)}
    let b_offset = ${b.indicesToOffset('b_indices')};
    var values: array<${output.type.value}, ${outputNumber}>;
    for (var k: u32 = 0u; k < uniforms.K; k = k + ${aComponents}) {
      ${calcResult()}
    }
    for (var i = 0u; i < ${outputNumber}u; i++) {
      var value = values[i];
      ${processBias}
      ${applyActivation}
      let cur_indices = ${output.type.indices}(batch, row + i, col);
      let offset = ${output.indicesToOffset('cur_indices')};
      ${output.setByOffset(`offset / ${components}`, 'value')};
    }
  }
  `;
      };
      return {
        name: 'MatMulNaive',
        shaderCache: {
          hint: `${activationAttributes.activation};${components};${aComponents};${outputNumber};${isChannelsLast}`,
          inputDependencies: hasBias ? ['rank', 'rank', 'rank'] : ['rank', 'rank']
        },
        getRunData: () => ({
          outputs: [{dims: outputShape, dataType: inputs[0].dataType}],
          dispatchGroup: {x: Math.ceil(outputSize / 64 /* workgroup size */)},
          programUniforms
        }),
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
  const N = outputShape[outputShape.length - 1];
  const K = context.inputs[0].dims[context.inputs[0].dims.length - 1];
  if (N < 8 && K < 8) {
    context.compute(createNaiveMatmulProgramInfo(context.inputs, {activation: ''}, outputShape));
  } else {
    context.compute(createMatmulProgramInfo(context.inputs, {activation: ''}, outputShape));
  }
};
