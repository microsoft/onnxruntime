// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {TensorView} from '../../tensor';
import {BroadcastUtil, ShapeUtil} from '../../util';
import {ComputeContext, GpuDataType, ProgramInfo, ProgramInfoLoader, ProgramMetadata} from '../types';

import {createIndicesHelper, ShaderHelper} from './common';

type BuiltinFunctionName = string;
type BinaryCustomExpression = (expressionA: string, expressionB: string) => string;
type BinaryFunctionCall = BuiltinFunctionName|BinaryCustomExpression|{
  scalar: BinaryCustomExpression;
  vector: BinaryCustomExpression;
};

const createBinaryOpProgramShader =
    (shaderHelper: ShaderHelper, dimsA: readonly number[], dimsB: readonly number[], dimsOutput: readonly number[],
     vectorize: boolean, doBroadcast: boolean, funcCall: BinaryFunctionCall, additionalImplementation?: string,
     typeA = 'f32', typeB = 'f32', typeOutput = 'f32') => {
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
            const idx = dimsOutput.length === 0 ? '0u' :
                (dimsOutput.length === 1)       ? '(*outputIndices)' :
                                                  `(*outputIndices)[${i + dimsOutput.length - dims.length}]`;
            offsets.push(`${strides[i]}u * (${idx} % ${dims[i]}u)`);
          }
          return offsets.length > 0 ? offsets.join('+') : '0u';
        };

        broadcastImpl = `
  ${outputIndicesHelper.o2iImpl}

  fn calcOffsetA(outputIndices: ptr<function, ${outputIndicesHelper.iType}>) -> u32 {
    return ${calcOffsetImpl(dimsA)};
  }

  fn calcOffsetB(outputIndices: ptr<function, ${outputIndicesHelper.iType}>) -> u32 {
    return ${calcOffsetImpl(dimsB)};
  }
  `;
      }

      let assignment: string;
      if (vectorize) {
        if (doBroadcast) {
          assignment = `
      ${outputIndicesHelper.indicesVariableDeclaration('outputIndices')}
      ${outputIndicesHelper.o2iCall('global_idx * 4u', 'outputIndices')}
      let offsetA = calcOffsetA(&outputIndices);
      let offsetB = calcOffsetB(&outputIndices);
      outputData[global_idx] = ${expressionVector('aData[offsetA / 4u]', 'bData[offsetB / 4u]')};`;
        } else {
          assignment = `outputData[global_idx] = ${expressionVector('aData[global_idx]', 'bData[global_idx]')};`;
        }
      } else {
        if (!doBroadcast) {
          throw new Error('no necessary to use scalar implementation for element-wise binary op implementation.');
        }
        const singleAssignment = (x: number) => {
          const expressionA = `aData[indexA${x}][componentA${x}]`;
          const expressionB = `bData[indexB${x}][componentB${x}]`;
          return `
      ${outputIndicesHelper.o2iCall(`global_idx * 4u + ${x}u`, 'outputIndices')}
      let offsetA${x} = calcOffsetA(&outputIndices);
      let offsetB${x} = calcOffsetB(&outputIndices);
      let indexA${x} = offsetA${x} / 4u;
      let indexB${x} = offsetB${x} / 4u;
      let componentA${x} = offsetA${x} % 4u;
      let componentB${x} = offsetB${x} % 4u;
      outputData[global_idx][${x}] = ${expressionScalar(expressionA, expressionB)};`;
        };

        assignment = `
      ${outputIndicesHelper.indicesVariableDeclaration('outputIndices')}
      ${singleAssignment(0)}
      ${singleAssignment(1)}
      ${singleAssignment(2)}
      ${singleAssignment(3)}`;
      }

      return `
  @group(0) @binding(0) var<storage, read> aData : array<vec4<${typeA}>>;
  @group(0) @binding(1) var<storage, read> bData : array<vec4<${typeB}>>;
  @group(0) @binding(2) var<storage, read_write> outputData : array<vec4<${typeOutput}>>;

  ${additionalImplementation ?? ''}
  ${broadcastImpl}

  ${shaderHelper.mainStart()}
    ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes(vecSize)}
    ${assignment}
  }`;
    };

const createBinaryOpProgramInfo =
    (metadata: ProgramMetadata, a: TensorView, b: TensorView, funcCall: BinaryFunctionCall,
     additionalImplementation?: string, outputDataType: number = a.dataType): ProgramInfo => {
      const isBroadcast = !ShapeUtil.areEqual(a.dims, b.dims);
      let outputShape = a.dims;
      let outputSize = ShapeUtil.size(a.dims);

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
        getShaderSource: (shaderHelper) => createBinaryOpProgramShader(
            shaderHelper, a.dims, b.dims, outputShape, vectorize, isBroadcast, funcCall, additionalImplementation),
        outputs: [{dims: outputShape, dataType: outputDataType, gpuDataType: GpuDataType.default}],
        dispatchGroup: () =>
            ({x: Math.ceil(outputSize / 64 /* workgroup size */ / (vectorize ? 4 : 1) /* vec size */)})
      };
    };

const createBinaryOpProgramInfoLoader =
    (inputs: readonly TensorView[], name: string, funcCall: BinaryFunctionCall, additionalImplementation?: string,
     cacheKey?: string): ProgramInfoLoader => {
      const metadata:
          ProgramMetadata = {name, inputTypes: [GpuDataType.default, GpuDataType.default], cacheHint: cacheKey};
      return {
        ...metadata,
        get: () => createBinaryOpProgramInfo(metadata, inputs[0], inputs[1], funcCall, additionalImplementation)
      };
    };

export const add = (context: ComputeContext): void => {
  context.compute(createBinaryOpProgramInfoLoader(context.inputs, 'Add', (a, b) => `${a}+${b}`));
};

export const div = (context: ComputeContext): void => {
  context.compute(createBinaryOpProgramInfoLoader(context.inputs, 'Div', (a, b) => `${a}/${b}`));
};

export const mul = (context: ComputeContext): void => {
  context.compute(createBinaryOpProgramInfoLoader(context.inputs, 'Mul', (a, b) => `${a}*${b}`));
};

export const pow = (context: ComputeContext): void => {
  context.compute(createBinaryOpProgramInfoLoader(
      context.inputs, 'Pow', ({scalar: (a, b) => `pow_f32(${a},${b})`, vector: (a, b) => `pow_vf32(${a},${b})`}), `
    fn pow_f32(a : f32, b : f32) -> f32 {
      if (b == 0.0) {
        return 1.0;
      } else if (a < 0.0 && b != floor(b)) {
        return pow(a, b); // NaN
      }
      return select(sign(a), 1.0, round(abs(b) % 2.0) != 1.0) * pow(abs(a), b);
    }
    fn pow_vf32(a : vec4<f32>, b : vec4<f32>) -> vec4<f32> {
      // TODO: implement vectorized pow
      return vec4<f32>(pow_f32(a.x, b.x), pow_f32(a.y, b.y), pow_f32(a.z, b.z), pow_f32(a.w, b.w));
    }
      `));
};

export const sub = (context: ComputeContext): void => {
  context.compute(createBinaryOpProgramInfoLoader(context.inputs, 'Sub', (a, b) => `${a}-${b}`));
};
