// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// import {AttributeWithCacheKey, createAttributeWithCacheKey} from '../../../attribute-with-cache-key';
import {WebGpuBackend} from '../../backend-webgpu';
import {Tensor} from '../../tensor';
import {BroadcastUtil, ShapeUtil} from '../../util';
import {GpuDataType, ProgramInfo, ProgramInfoLoader, ProgramMetadata} from '../types';

import {createIndicesHelper, WORKGROUP_SIZE} from './common';

type BuiltinFunctionName = string;
type BinaryCustomExpression = (expressionA: string, expressionB: string) => string;
type BinaryFunctionCall = BuiltinFunctionName|BinaryCustomExpression|{
  scalar: BinaryCustomExpression;
  vector: BinaryCustomExpression;
};

const createBinaryOpProgramShader =
    (dimsA: readonly number[], dimsB: readonly number[], dimsOutput: readonly number[], vectorize: boolean,
     doBroadcast: boolean, funcCall: BinaryFunctionCall, additionalImplementation?: string, typeA = 'f32',
     typeB = 'f32', typeOutput = 'f32') => {
      const outputSize = ShapeUtil.size(dimsOutput);
      const vecSize = Math.ceil(outputSize / 4);

      let expressionScalar: BinaryCustomExpression;
      let expressionVector: BinaryCustomExpression;
      if (typeof funcCall === 'string') {
        expressionScalar = expressionVector = (a, b) => `${funcCall}((${a}),(${b}))`;
      } else if (typeof funcCall === 'function') {
        expressionScalar = expressionVector = funcCall;
      } else {
        expressionScalar = funcCall.scalar;
        expressionVector = funcCall.vector;
      }

      let broadcastImpl = '';
      const outputIndicesHelper = createIndicesHelper('output', dimsOutput);
      if (doBroadcast) {
        const calcOffsetImpl = (dims: readonly number[]) => {
          const strides = ShapeUtil.computeStrides(dims);
          const offsets: string[] = [];
          for (let i = dims.length - 1; i >= 0; i--) {
            offsets.push(`${strides[i]}u * ((*outputIndices)[${i + dimsOutput.length - dims.length}] % ${dims[i]}u)`);
          }
          return offsets.length > 0 ? offsets.join('+') : '0u';
        };

        broadcastImpl = `
  ${outputIndicesHelper.o2iImpl}

  fn calcOffsetA(outputIndices: ptr<function, array<u32, ${dimsOutput.length}>>) -> u32 {
    return ${calcOffsetImpl(dimsA)};
  }

  fn calcOffsetB(outputIndices: ptr<function, array<u32, ${dimsOutput.length}>>) -> u32 {
    return ${calcOffsetImpl(dimsB)};
  }
  `;
      }

      let assignment: string;
      if (vectorize) {
        if (doBroadcast) {
          assignment = `
      ${outputIndicesHelper.indicesVariableDeclaration('outputIndices')}
      ${outputIndicesHelper.o2iCall('global_id.x * 4u', 'outputIndices')}
      let offsetA = calcOffsetA(&outputIndices);
      let offsetB = calcOffsetB(&outputIndices);
      outputData[global_id.x] = ${expressionVector('aData[offsetA / 4u]', 'bData[offsetB / 4u]')};`;
        } else {
          assignment = `outputData[global_id.x] = ${expressionVector('aData[global_id.x]', 'bData[global_id.x]')};`;
        }
      } else {
        if (!doBroadcast) {
          throw new Error('no necessary to use scalar implementation for element-wise binary op implementation.');
        }
        const singleAssignment = (x: number) => {
          const expressionA = `aData[indexA${x}][componentA${x}]`;
          const expressionB = `bData[indexB${x}][componentB${x}]`;
          return `
      ${outputIndicesHelper.o2iCall(`global_id.x * 4u + ${x}u`, 'outputIndices')}
      let offsetA${x} = calcOffsetA(&outputIndices);
      let offsetB${x} = calcOffsetB(&outputIndices);
      let indexA${x} = offsetA${x} / 4u;
      let indexB${x} = offsetB${x} / 4u;
      let componentA${x} = offsetA${x} % 4u;
      let componentB${x} = offsetB${x} % 4u;
      outputData[global_id.x][${x}] = ${expressionScalar(expressionA, expressionB)};`;
        };

        assignment = `
      ${outputIndicesHelper.indicesVariableDeclaration('outputIndices')}
      ${singleAssignment(0)}
      ${singleAssignment(1)}
      ${singleAssignment(2)}
      ${singleAssignment(3)}`;
      }

      return `
  const WORKGROUP_SIZE: u32 = ${WORKGROUP_SIZE}u;

  @group(0) @binding(0) var<storage, read> aData : array<vec4<${typeA}>>;
  @group(0) @binding(1) var<storage, read> bData : array<vec4<${typeB}>>;
  @group(0) @binding(2) var<storage, read_write> outputData : array<vec4<${typeOutput}>>;

  ${additionalImplementation ?? ''}
  ${broadcastImpl}

  @compute @workgroup_size(WORKGROUP_SIZE)
  fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {

    // Guard against out-of-bounds work group sizes
    if (global_id.x >= ${vecSize}u) {
      return;
    }

    ${assignment}
  }`;
    };

const createBinaryOpProgramInfo =
    (metadata: ProgramMetadata, a: Tensor, b: Tensor, funcCall: BinaryFunctionCall, additionalImplementation?: string,
     outputTensorType: Tensor.DataType = a.type): ProgramInfo => {
      const isBroadcast = !ShapeUtil.areEqual(a.dims, b.dims);
      let outputShape = a.dims;
      let outputSize = a.size;

      let vectorize = false;

      // TODO: deal with zero-sized tensors (eg. dims=[1,0])

      if (isBroadcast) {
        const calculatedShape = BroadcastUtil.calcShape(a.dims, b.dims, false);
        if (!calculatedShape) {
          throw new Error('Can\'t perform binary op on the given tensors');
        }
        outputShape = calculatedShape;
        outputSize = ShapeUtil.size(outputShape);

        // check whether vectorize can be enabled
        let sharedDimension = 1;
        for (let i = 0; i < outputShape.length; i++) {
          const dimA = a.dims[a.dims.length - i] ?? 1;
          const dimB = b.dims[b.dims.length - i] ?? 1;
          if (dimA === dimB) {
            sharedDimension *= dimA;
          } else {
            break;
          }
        }
        if (sharedDimension % 4 === 0) {
          vectorize = true;
        }


      } else {
        // element-wise
        vectorize = true;
      }

      return {
        ...metadata,
        shaderSource: createBinaryOpProgramShader(
            a.dims, b.dims, outputShape, vectorize, isBroadcast, funcCall, additionalImplementation),
        outputs: [{dims: outputShape, type: outputTensorType, gpuDataType: GpuDataType.default}],
        dispatchGroup: () =>
            ({x: Math.ceil(outputSize / 64 /* workgroup size */ / (vectorize ? 4 : 1) /* vec size */)})
      };
    };

const createBinaryOpProgramInfoLoader =
    (inputs: Tensor[], name: string, funcCall: BinaryFunctionCall, additionalImplementation?: string,
     cacheKey?: string): ProgramInfoLoader => {
      const metadata:
          ProgramMetadata = {name, inputTypes: [GpuDataType.default, GpuDataType.default], cacheHint: cacheKey};
      return {
        ...metadata,
        get: () => createBinaryOpProgramInfo(metadata, inputs[0], inputs[1], funcCall, additionalImplementation)
      };
    };

export const add = async(backend: WebGpuBackend, inputs: Tensor[]): Promise<Tensor[]> =>
    backend.run(createBinaryOpProgramInfoLoader(inputs, 'Add', (a, b) => `${a}+${b}`), inputs);

// export const and = (backend: WebGLInferenceHandler, inputs: Tensor[]):
//     Tensor[] => [backend.run(createBinaryProgramInfoLoader(backend, inputs, glslAnd(), 'bool'), inputs)];

export const div = async(backend: WebGpuBackend, inputs: Tensor[]): Promise<Tensor[]> =>
    backend.run(createBinaryOpProgramInfoLoader(inputs, 'Div', (a, b) => `${a}/${b}`), inputs);

// export const equal = (backend: WebGLInferenceHandler, inputs: Tensor[]):
//     Tensor[] => [backend.run(createBinaryProgramInfoLoader(backend, inputs, glslEqual(), 'bool'), inputs)];

// export const greater = (backend: WebGLInferenceHandler, inputs: Tensor[]):
//     Tensor[] => [backend.run(createBinaryProgramInfoLoader(backend, inputs, glslGreater(), 'bool'), inputs)];

// export const less = (backend: WebGLInferenceHandler, inputs: Tensor[]):
//     Tensor[] => [backend.run(createBinaryProgramInfoLoader(backend, inputs, glslLess(), 'bool'), inputs)];

export const mul = async(backend: WebGpuBackend, inputs: Tensor[]): Promise<Tensor[]> =>
    backend.run(createBinaryOpProgramInfoLoader(inputs, 'Mul', (a, b) => `${a}*${b}`), inputs);

// export const or = (backend: WebGLInferenceHandler, inputs: Tensor[]):
//     Tensor[] => [backend.run(createBinaryProgramInfoLoader(backend, inputs, glslOr(), 'bool'), inputs)];

export const pow = async(backend: WebGpuBackend, inputs: Tensor[]): Promise<Tensor[]> =>
    backend.run(createBinaryOpProgramInfoLoader(inputs, 'Pow', 'pow'), inputs);

// export const pRelu = (backend: WebGLInferenceHandler, inputs: Tensor[]):
//     Tensor[] => [backend.run(createBinaryProgramInfoLoader(backend, inputs, glslPRelu()), inputs)];

export const sub = async(backend: WebGpuBackend, inputs: Tensor[]): Promise<Tensor[]> =>
    backend.run(createBinaryOpProgramInfoLoader(inputs, 'Sub', (a, b) => `${a}-${b}`), inputs);

// export const xor = (backend: WebGLInferenceHandler, inputs: Tensor[]):
//     Tensor[] => [backend.run(createBinaryProgramInfoLoader(backend, inputs, glslXor(), 'bool'), inputs)];
