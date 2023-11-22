// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {tensorDataTypeEnumToString} from '../../../wasm-common';
import {TensorView} from '../../tensor-view';
import {GemmUtil, ShapeUtil} from '../../util';
import {ComputeContext, ProgramInfo, ProgramInputTensorInfoDependency, ProgramUniform} from '../types';

import {createTensorShapeVariables, getElementAt, inputVariable, outputVariable, ShaderHelper, UniformDataElementType, UniformsArrayType} from './common';

const validateInputs = (inputs: readonly TensorView[]): void => {
  if (!inputs) {
    throw new Error('Input is missing');
  }
  if (inputs.length < 2 || inputs.length > 3) {
    throw new Error('Invaid input number.');
  }

  // 'C' can be of dimensionality 0, 1 or 2 only
  if (inputs.length === 3 && inputs[2].dims.length > 2) {
    throw new Error('Invalid input shape of C');
  }

  if ((inputs[0].dataType !== inputs[1].dataType) ||
      (inputs.length === 3 && inputs[0].dataType !== inputs[2].dataType)) {
    throw new Error('Input types are mismatched');
  }
};

interface GemmAttributes {
  transA: boolean;
  transB: boolean;
  alpha: number;
  beta: number;
}

const calculateC = (m: number, n: number, dims: readonly number[]): [string, boolean, boolean] => {
  let calculate = '0u';
  if (dims.length !== 0) {
    const broadcastM = (dims.length === 1 && m !== 1) || (dims.length === 2 && dims[0] !== m);
    const broadcastN = dims[dims.length - 1] !== n;

    if (!broadcastM) {
      calculate += `+ m * ${getElementAt('uniforms.c_shape', dims.length - 1, dims.length)}`;
    }
    if (!broadcastN) {
      calculate += '+ n';
    }
    return [`value += uniforms.beta * c[${calculate}];`, broadcastM, broadcastN];
  }
  return [`value += uniforms.beta * c[${calculate}];`, false, false];
};

const createGemmProgramInfo = (inputs: readonly TensorView[], attributes: GemmAttributes): ProgramInfo => {
  const aShape = inputs[0].dims.slice();
  const bShape = inputs[1].dims.slice();
  const [M, N, K] = GemmUtil.getShapeOfGemmResult(
      aShape, attributes.transA, bShape, attributes.transB, inputs.length === 3 ? inputs[2].dims : undefined);
  const outputShape = [M, N];
  if (!outputShape) {
    throw new Error('Can\'t use gemm on the given tensors');
  }
  const outputSize = ShapeUtil.size(outputShape);
  let line = '';
  if (attributes.transA && attributes.transB) {
    line = 'value += a[k * uniforms.M + m] * b[n * uniforms.K + k];';
  } else if (attributes.transA && !attributes.transB) {
    line = 'value += a[k * uniforms.M + m] * b[k * uniforms.N + n];';
  } else if (!attributes.transA && attributes.transB) {
    line = 'value += a[m * uniforms.K + k] * b[n * uniforms.K + k];';
  } else if (!attributes.transA && !attributes.transB) {
    line = 'value += a[m * uniforms.K + k] * b[k * uniforms.N + n];';
  }

  const calculateAlpha = attributes.alpha === 1 ? '' : 'value *= uniforms.alpha;';
  const [calculateCSnippet, broadcastM, broadcastN] =
      inputs.length === 3 ? calculateC(M, N, inputs[2].dims) : ['', false, false];

  const a = inputVariable('a', inputs[0].dataType, inputs[0].dims.length);
  const b = inputVariable('b', inputs[1].dataType, inputs[1].dims.length);
  const dataType = a.type.value;

  const variables = [a, b];
  const tensorDataType = tensorDataTypeEnumToString(inputs[0].dataType) as ProgramUniform['type'];
  const programUniforms: ProgramUniform[] = [
    {type: 'uint32', data: outputSize}, {type: 'uint32', data: M}, {type: 'uint32', data: N}, {type: 'uint32', data: K},
    {type: tensorDataType, data: attributes.alpha}, {type: tensorDataType, data: attributes.beta},
    ...createTensorShapeVariables(inputs[0].dims), ...createTensorShapeVariables(inputs[1].dims)
  ];

  const inputDependencies: ProgramInputTensorInfoDependency[] = ['rank', 'rank'];
  if (inputs.length === 3) {
    variables.push(inputVariable('c', inputs[2].dataType, inputs[2].dims.length));
    programUniforms.push(...createTensorShapeVariables(inputs[2].dims));
    inputDependencies.push('rank');
  }
  const output = outputVariable('output', inputs[0].dataType, outputShape.length);
  variables.push(output);
  programUniforms.push(...createTensorShapeVariables(outputShape));

  const uniforms: UniformsArrayType = [
    {name: 'outputSize', type: 'u32'}, {name: 'M', type: 'u32'}, {name: 'N', type: 'u32'}, {name: 'K', type: 'u32'},
    {name: 'alpha', type: dataType as UniformDataElementType}, {name: 'beta', type: dataType as UniformDataElementType}
  ];

  const getShaderSource = (shaderHelper: ShaderHelper) => `
  ${shaderHelper.registerUniforms(uniforms).declareVariables(...variables)}

  ${shaderHelper.mainStart()}
    ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes('uniforms.outputSize')}

    let m = global_id.x / uniforms.N;
    let n = global_id.x % uniforms.N;

    var value = ${dataType}(0);
    for (var k: u32 = 0u; k < uniforms.K; k++) {
      ${line}
    }

    ${calculateAlpha}
    ${calculateCSnippet}
    output[global_id.x] = value;

  }`;
  return {
    name: 'Gemm',
    shaderCache: {
      hint: `${attributes.transA};${attributes.transB};${attributes.alpha === 1};${broadcastM};${broadcastN}`,
      inputDependencies
    },
    getRunData: () => ({
      outputs: [{dims: outputShape, dataType: inputs[0].dataType}],
      dispatchGroup: {x: Math.ceil(outputSize / 64 /* workgroup size */)},
      programUniforms
    }),
    getShaderSource,
  };
};

export const gemm = (context: ComputeContext, attributes: GemmAttributes): void => {
  validateInputs(context.inputs);
  context.compute(createGemmProgramInfo(context.inputs, attributes));
};
