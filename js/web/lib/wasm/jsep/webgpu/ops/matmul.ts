// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {TensorView} from '../../tensor-view';
import {BroadcastUtil, ShapeUtil} from '../../util';
import {ComputeContext, ProgramInfo} from '../types';

import {getBroadcastDims, IndicesHelper, inputVariable, outputVariable, ShaderHelper,} from './common';
import {getActicationSnippet, InternalActivationAttributes} from './fuse-utils';

const getMaxComponents = (size: number): 1|2|3|4 => {
  if (size % 4 === 0) {
    return 4;
  } else if (size % 2 === 0) {
    return 2;
  }
  return 1;
};
export const createMatmulLWGProgramInfo =
    (inputs: readonly TensorView[], activationAttributes: InternalActivationAttributes, outputShape: readonly number[],
     reshapedOutputShape?: readonly number[],
     isChannelsLast = false /* only used for conv2dByMatMul*/): ProgramInfo => {
      const aShape = inputs[0].dims;
      const bShape = inputs[1].dims;

      const {activationFunction, applyActivation} = getActicationSnippet(activationAttributes);

      const M = aShape[aShape.length - 2];
      const N = bShape[bShape.length - 1];
      const K = aShape[aShape.length - 1];
      const outerDims = reshapedOutputShape ? reshapedOutputShape.slice(0, -2) : outputShape.slice(0, -2);
      const batchSize = ShapeUtil.size(outerDims);
      const components = getMaxComponents(N);
      const aComponents = getMaxComponents(K);
      const tileN = N < 32 ? N : 32;
      const tileM = M < 32 ? M : 32;
      const tileK = K < 32 ? K : 32;
      const workgroupSize = 64;
      // The output number of each thread.
      const outputNumber = Math.ceil(tileM / Math.floor(workgroupSize / (tileN / components)));
      if (workgroupSize < (tileN / components)) {
        throw new Error(
            `workgroupSize ${workgroupSize} must be larger than or equal to tileN / components ${tileN / components}`);
      }
      // The virtualXXX makes sure that one tile of data has the same batch.
      const virtualM = Math.ceil(M / tileM) * tileM;
      const virtualN = Math.ceil(N / tileN) * tileN;
      const numWorkgroups = ShapeUtil.size([batchSize, virtualM, virtualN]) / tileM / tileN;
      const a = inputVariable('a', inputs[0].dataType, aShape, aComponents);
      const b = inputVariable('b', inputs[1].dataType, bShape, components);
      const output = outputVariable('output', inputs[0].dataType, [batchSize, M, N], components);
      const inputVariables = [a, b];
      const hasBias = inputs.length > 2;
      let processBias = '';
      if (hasBias) {
        const biasComponents = isChannelsLast ? components : 1;
        inputVariables.push(inputVariable('bias', inputs[2].dataType, inputs[2].dims, biasComponents));
        processBias = `${
            isChannelsLast ? `value += bias[col / ${biasComponents}];` : `value += ${output.type.value}(bias[row]);`}`;
      }
      const outerDimsA = aShape.slice(0, -2);
      const outerDimsB = bShape.slice(0, -2);
      const batchDims = inputVariable('batchDims', inputs[0].dataType, outerDims);
      const broadCastADims = getBroadcastDims(outerDimsA, outerDims);
      const broadCastBDims = getBroadcastDims(outerDimsB, outerDims);
      const getIndices = (variable: IndicesHelper, broadCastDims: number[]) => {
        const rank = variable.rank;
        const name = variable.name;
        if (rank === 2) {
          return `var ${name}Indices = ${variable.type.indices}(0u, 0u);`;
        }
        const batchRank = outerDims.length;
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
            let bData${i} = mm_Bsub[k + ${i}][localCol];`;
        }
        for (let i = 0; i < outputNumber; i++) {
          calcStr += `aData = mm_Asub[localRow + ${i}][k / ${aComponents}];`;

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
  fn mm_readA(batchOffset: u32, row: u32, col: u32) -> ${a.type.value} {
    var value = ${a.type.value}(0.0);
    if(row < M && col < K)
    {
      value = a[(batchOffset + row * K + col) / ${aComponents}];
    }
    return value;
  }
  fn mm_readB(batchOffset: u32, row: u32, col: u32) -> ${b.type.value} {
    var value = ${b.type.value}(0.0);
    if(row < K && col < N)
    {
      value = b[(batchOffset + row * N + col) / ${components}];
    }
    return value;
  }
  fn mm_write(batch: u32, row : u32, col : u32, valueIn : ${output.type.value}) {
    if (row < M && col < N) {
      var value = valueIn;
      ${processBias}
      ${applyActivation}
      let curIndices = ${output.type.indices}(batch, row, col);
      let offset = ${output.indicesToOffset('curIndices')};
      ${output.setByOffset(`offset / ${components}`, 'value')};
    }
  }
  var<workgroup> mm_Asub: array<array<${a.type.storage}, ${tileK / aComponents}>, ${tileM}>;
  var<workgroup> mm_Bsub: array<array<${b.type.storage}, ${tileN / components}>, ${tileK}>;
  ${shaderHelper.mainStart(workgroupSize)}
    let virtualGlobalId = workgroup_id.z * numWorkgroups.x * numWorkgroups.y +
        workgroup_id.y * numWorkgroups.x + workgroup_id.x;
    let tileColStart = (virtualGlobalId % ${virtualN / tileN}u) * ${tileN}u;
    var index1 = virtualGlobalId / ${virtualN / tileN}u;
    let stride1 = ${virtualM / tileM}u;
    let tileRowStart = (index1 % stride1) * ${tileM}u;
    let batch = index1 / stride1;

    // The current thread location in a tile.
    let localIndex = local_id.x;
    let localRow = localIndex / ${tileN / components} * ${outputNumber};
    let localCol = localIndex % ${tileN / components};

    ${outputShape.length === 2 ? '' : `let batchIndices = ${batchDims.offsetToIndices('batch')};`}
    ${getIndices(a, broadCastADims)}
    let offsetA = ${a.indicesToOffset('aIndices')};
    ${getIndices(b, broadCastBDims)}
    let offsetB = ${b.indicesToOffset('bIndices')};
    var values: array<${output.type.value}, ${outputNumber}>;

    let numTiles = (K - 1u) / ${tileK} + 1u;
    var kStart = 0u;
    // Loop over shared dimension.
    for (var t = 0u; t < numTiles; t = t + 1u) {
      // Load one tile of A into local memory.
      for (var tIndex = localIndex; tIndex < ${tileM * tileK / aComponents}; tIndex += ${workgroupSize}) {
          let inputRow = tIndex / ${tileK / aComponents};
          let inputCol = tIndex % ${tileK / aComponents};

          mm_Asub[inputRow][inputCol] = mm_readA(offsetA, tileRowStart + inputRow, kStart + inputCol * ${aComponents});
      }

      // Load one tile of B into local memory.
      for (var tIndex = localIndex; tIndex < ${tileK * tileN / components}; tIndex += ${workgroupSize}) {
          let inputRow = tIndex / ${tileN / components};
          let inputCol = tIndex % ${tileN / components};
          mm_Bsub[inputRow][inputCol] = mm_readB(offsetB, kStart + inputRow, tileColStart + inputCol * ${components});
      }
      kStart = kStart + ${tileK};
      workgroupBarrier();

      // Compute values for a single thread.
      for (var k = 0; k < ${tileK}; k = k + ${aComponents}) {
        ${calcResult()}
      }
      workgroupBarrier();
    }

    let globalCol = tileColStart + localCol * ${components};
    var globalRow = tileRowStart + localRow;
    for (var i = 0u; i < ${outputNumber}u; i++) {
      if (localRow + i >= ${tileM})
      {
        return;
      }
      mm_write(batch, globalRow + i, globalCol, values[i]);
    }
  }
  ${batchDims.impl()}
  `;
      return {
        name: 'MatMulLinearWG',
        shaderCache: {hint: activationAttributes.activationCacheKey},
        getRunData: () =>
            ({outputs: [{dims: outputShape, dataType: inputs[0].dataType}], dispatchGroup: {x: numWorkgroups}}),
        getShaderSource,
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
  context.compute(createMatmulLWGProgramInfo(context.inputs, {activation: '', activationCacheKey: ''}, outputShape));
};
