// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {DataType} from '../../../wasm-common';
import {TensorView} from '../../tensor-view';
import {BroadcastUtil, ShapeUtil} from '../../util';
import {ComputeContext, ProgramInfo} from '../types';

import {inputVariable, outputVariable, ShaderHelper} from './common';

type BuiltinFunctionName = string;
type BinaryCustomExpression = (expressionA: string, expressionB: string) => string;
type BinaryFunctionCall = BuiltinFunctionName|BinaryCustomExpression|{
  scalar: BinaryCustomExpression;
  vector: BinaryCustomExpression;
};

const createBinaryOpProgramShader =
    (shaderHelper: ShaderHelper, dimsA: readonly number[], dimsB: readonly number[], dimsOutput: readonly number[],
     vectorize: boolean, doBroadcast: boolean, funcCall: BinaryFunctionCall, typeA: number, typeB: number,
     typeOutput: number, additionalImplementation?: string) => {
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
      const output = outputVariable('outputData', typeOutput, dimsOutput, 4);
      const a = inputVariable('aData', typeA, dimsA, 4);
      const b = inputVariable('bData', typeB, dimsB, 4);
      if (doBroadcast) {
        const calcOffsetImpl = (dims: readonly number[]) => {
          const strides = ShapeUtil.computeStrides(dims);
          const offsets: string[] = [];
          for (let i = dims.length - 1; i >= 0; i--) {
            const idx = output.indicesGet('outputIndices', i + dimsOutput.length - dims.length);
            offsets.push(`${strides[i]}u * (${idx} % ${dims[i]}u)`);
          }
          return offsets.length > 0 ? offsets.join('+') : '0u';
        };

        broadcastImpl = `
          fn calcOffsetA(outputIndices: ${output.type.indices}) -> u32 {
            return ${calcOffsetImpl(dimsA)};
          }

          fn calcOffsetB(outputIndices: ${output.type.indices}) -> u32 {
            return ${calcOffsetImpl(dimsB)};
          }
        `;
      }

      let assignment: string;
      if (vectorize) {
        if (doBroadcast) {
          const isAOneElement = ShapeUtil.size(dimsA) === 1;
          const isBOneElement = ShapeUtil.size(dimsB) === 1;
          if (isAOneElement || isBOneElement) {
            assignment = output.setByOffset(
                'global_idx',
                expressionVector(
                    isAOneElement ? `${a.type.value}(${a.getByOffset('0')}.x)` : a.getByOffset('global_idx'),
                    isBOneElement ? `${b.type.value}(${b.getByOffset('0')}.x)` : b.getByOffset('global_idx')));
          } else {
            assignment = `
            let outputIndices = ${output.offsetToIndices('global_idx * 4u')};
            let offsetA = calcOffsetA(outputIndices);
            let offsetB = calcOffsetB(outputIndices);
            ${
                output.setByOffset(
                    'global_idx', expressionVector(a.getByOffset('offsetA / 4u'), b.getByOffset('offsetB / 4u')))}
          `;
          }
        } else {
          assignment = output.setByOffset(
              'global_idx', expressionVector(a.getByOffset('global_idx'), b.getByOffset('global_idx')));
        }
      } else {
        if (!doBroadcast) {
          throw new Error('no necessary to use scalar implementation for element-wise binary op implementation.');
        }

        const singleAssignment = (resStr: string, x: number, typeCast = '') => {
          const expressionA = `aData[indexA${x}][componentA${x}]`;
          const expressionB = `bData[indexB${x}][componentB${x}]`;
          return `
            let outputIndices${x} = ${output.offsetToIndices(`global_idx * 4u + ${x}u`)};
            let offsetA${x} = calcOffsetA(outputIndices${x});
            let offsetB${x} = calcOffsetB(outputIndices${x});
            let indexA${x} = offsetA${x} / 4u;
            let indexB${x} = offsetB${x} / 4u;
            let componentA${x} = offsetA${x} % 4u;
            let componentB${x} = offsetB${x} % 4u;
            ${resStr}[${x}] = ${typeCast}(${expressionScalar(expressionA, expressionB)});
          `;
        };
        if (typeOutput === DataType.bool) {
          assignment = `
            var data = vec4<u32>(0);
            ${singleAssignment('data', 0, 'u32')}
            ${singleAssignment('data', 1, 'u32')}
            ${singleAssignment('data', 2, 'u32')}
            ${singleAssignment('data', 3, 'u32')}
            outputData[global_idx] = dot(vec4<u32>(0x1, 0x100, 0x10000, 0x1000000), vec4<u32>(data));`;
        } else {
          assignment = `
            ${singleAssignment('outputData[global_idx]', 0)}
            ${singleAssignment('outputData[global_idx]', 1)}
            ${singleAssignment('outputData[global_idx]', 2)}
            ${singleAssignment('outputData[global_idx]', 3)}
          `;
        }
      }

      return `
        ${shaderHelper.declareVariables(a, b, output)}

        ${additionalImplementation ?? ''}
        ${broadcastImpl}

        ${shaderHelper.mainStart()}
        ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes(vecSize)}
        ${assignment}
      }`;
    };

const createBinaryOpProgramInfo =
    (name: string, cacheKey: string, a: TensorView, b: TensorView, funcCall: BinaryFunctionCall,
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
        const isAOneElement = ShapeUtil.size(a.dims) === 1;
        const isBOneElement = ShapeUtil.size(b.dims) === 1;

        // check whether vectorize can be enabled
        let sharedDimension = 1;
        for (let i = 1; i < outputShape.length; i++) {
          const dimA = a.dims[a.dims.length - i] ?? 1;
          const dimB = b.dims[b.dims.length - i] ?? 1;
          if (dimA === dimB) {
            sharedDimension *= dimA;
          } else {
            break;
          }
        }
        if (sharedDimension % 4 === 0 || isAOneElement || isBOneElement) {
          vectorize = true;
        }
      } else {
        // element-wise
        vectorize = true;
      }

      return {
        name,
        shaderCache: {hint: cacheKey},
        getShaderSource: (shaderHelper) => createBinaryOpProgramShader(
            shaderHelper, a.dims, b.dims, outputShape, vectorize, isBroadcast, funcCall, a.dataType, b.dataType,
            outputDataType, additionalImplementation),
        getRunData: () => ({
          outputs: [{dims: outputShape, dataType: outputDataType}],
          dispatchGroup: {x: Math.ceil(outputSize / 64 /* workgroup size */ / 4 /* component size */)}
        }),
      };
    };

const runBinaryOp =
    (context: ComputeContext, name: string, funcCall: BinaryFunctionCall, additionalImplementation?: string,
     cacheKey?: string, outputDataType?: number): void => {
      context.compute(createBinaryOpProgramInfo(
          name, cacheKey ?? '', context.inputs[0], context.inputs[1], funcCall, additionalImplementation,
          outputDataType));
    };

export const add = (context: ComputeContext): void => {
  runBinaryOp(context, 'Add', (a, b) => `${a}+${b}`);
};

export const div = (context: ComputeContext): void => {
  runBinaryOp(context, 'Div', (a, b) => `${a}/${b}`);
};

export const equal = (context: ComputeContext): void => {
  runBinaryOp(
      context, 'Equal', ({scalar: (a, b) => `u32(${a}==${b})`, vector: (a, b) => `vec4<u32>(${a}==${b})`}), undefined,
      undefined, DataType.bool);
};

export const mul = (context: ComputeContext): void => {
  runBinaryOp(context, 'Mul', (a, b) => `${a}*${b}`);
};

export const pow = (context: ComputeContext): void => {
  const type = inputVariable('input', context.inputs[0].dataType, context.inputs[0].dims).type.value;
  const roundStr = type === 'i32' ? 'round' : '';
  runBinaryOp(
      context, 'Pow', ({scalar: (a, b) => `pow_custom(${a},${b})`, vector: (a, b) => `pow_vector_custom(${a},${b})`}),
      `
    fn pow_custom(a : ${type}, b : ${type}) -> ${type} {
      if (b == ${type}(0.0)) {
        return ${type}(1.0);
      } else if (a < ${type}(0.0) && f32(b) != floor(f32(b))) {
        return ${type}(pow(f32(a), f32(b))); // NaN
      }
      return select(sign(a), ${type}(1.0), round(f32(abs(b) % ${type}(2.0))) != 1.0) * ${type}(${
          roundStr}(pow(f32(abs(a)), f32(b))));
    }
    fn pow_vector_custom(a : vec4<${type}>, b : vec4<${type}>) -> vec4<${type}> {
      // TODO: implement vectorized pow
      return vec4<${type}>(pow_custom(a.x, b.x), pow_custom(a.y, b.y), pow_custom(a.z, b.z), pow_custom(a.w, b.w));
    }
      `);
};

export const sub = (context: ComputeContext): void => {
  runBinaryOp(context, 'Sub', (a, b) => `${a}-${b}`);
};

export const greater = (context: ComputeContext): void => {
  runBinaryOp(
      context, 'Greater', ({scalar: (a, b) => `u32(${a}>${b})`, vector: (a, b) => `vec4<u32>(${a}>${b})`}), undefined,
      undefined, DataType.bool);
};

export const less = (context: ComputeContext): void => {
  runBinaryOp(
      context, 'Less', ({scalar: (a, b) => `u32(${a}<${b})`, vector: (a, b) => `vec4<u32>(${a}<${b})`}), undefined,
      undefined, DataType.bool);
};

export const greaterOrEqual = (context: ComputeContext): void => {
  runBinaryOp(
      context, 'GreaterOrEqual', ({scalar: (a, b) => `u32(${a}>=${b})`, vector: (a, b) => `vec4<u32>(${a}>=${b})`}),
      undefined, undefined, DataType.bool);
};

export const lessOrEqual = (context: ComputeContext): void => {
  runBinaryOp(
      context, 'LessOrEqual', ({scalar: (a, b) => `u32(${a}<=${b})`, vector: (a, b) => `vec4<u32>(${a}<=${b})`}),
      undefined, undefined, DataType.bool);
};
