// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {DataType} from '../../../wasm-common';
import {TensorView} from '../../tensor';
import {ShapeUtil} from '../../util';
import {ComputeContext} from '../types';

import {BinaryCustomExpression, BinaryFunctionCall, calcOffsetImpl, createOpProgramInfoLoader, fourAssignment, getIndexComponent} from './binary-like-util';
import {inputVariable, outputVariable, ShaderHelper} from './common';

const createOpProgramShader =
    (shaderHelper: ShaderHelper, inputs: readonly TensorView[], dimsOutput: readonly number[], vectorize: boolean,
     doBroadcast: boolean, funcCall: BinaryFunctionCall, typeOutput: number, additionalImplementation?: string) => {
      const typeA = inputs[0].dataType;
      const typeB = inputs[1].dataType;
      const dimsA = inputs[0].dims;
      const dimsB = inputs[1].dims;
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
        broadcastImpl = `
          ${calcOffsetImpl('A', dimsA, output, dimsOutput)}
          ${calcOffsetImpl('B', dimsB, output, dimsOutput)}
        `;
      }

      let assignment: string;
      if (vectorize) {
        if (doBroadcast) {
          assignment = `
            let outputIndices = ${output.offsetToIndices('global_idx * 4u')};
            let offsetA = calcOffsetA(outputIndices);
            let offsetB = calcOffsetB(outputIndices);
            ${
              output.setByOffset(
                  'global_idx', expressionVector(a.getByOffset('offsetA / 4u'), b.getByOffset('offsetB / 4u')))}
          `;
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
            ${getIndexComponent('A', x)}
            ${getIndexComponent('B', x)}
            ${resStr}[${x}] = ${typeCast}(${expressionScalar(expressionA, expressionB)});
          `;
        };
        assignment = fourAssignment(singleAssignment, typeOutput);
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

export const add = (context: ComputeContext): void => {
  context.compute(createOpProgramInfoLoader(context.inputs, 'Add', (a, b) => `${a}+${b}`, createOpProgramShader));
};

export const div = (context: ComputeContext): void => {
  context.compute(createOpProgramInfoLoader(context.inputs, 'Div', (a, b) => `${a}/${b}`, createOpProgramShader));
};

export const equal = (context: ComputeContext): void => {
  context.compute(createOpProgramInfoLoader(
      context.inputs, 'Equal', ({scalar: (a, b) => `u32(${a}==${b})`, vector: (a, b) => `vec4<u32>(${a}==${b})`}),
      createOpProgramShader, undefined, undefined, DataType.bool));
};

export const mul = (context: ComputeContext): void => {
  context.compute(createOpProgramInfoLoader(context.inputs, 'Mul', (a, b) => `${a}*${b}`, createOpProgramShader));
};

export const pow = (context: ComputeContext): void => {
  const type = inputVariable('input', context.inputs[0].dataType, context.inputs[0].dims).type.value;
  const roundStr = type === 'i32' ? 'round' : '';
  context.compute(createOpProgramInfoLoader(
      context.inputs, 'Pow',
      ({scalar: (a, b) => `pow_custom(${a},${b})`, vector: (a, b) => `pow_vector_custom(${a},${b})`}),
      createOpProgramShader,
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
      `));
};

export const sub = (context: ComputeContext): void => {
  context.compute(createOpProgramInfoLoader(context.inputs, 'Sub', (a, b) => `${a}-${b}`, createOpProgramShader));
};

export const greater = (context: ComputeContext): void => {
  context.compute(createOpProgramInfoLoader(
      context.inputs, 'Greater', ({scalar: (a, b) => `u32(${a}>${b})`, vector: (a, b) => `vec4<u32>(${a}>${b})`}),
      createOpProgramShader, undefined, undefined, DataType.bool));
};

export const less = (context: ComputeContext): void => {
  context.compute(createOpProgramInfoLoader(
      context.inputs, 'Less', ({scalar: (a, b) => `u32(${a}<${b})`, vector: (a, b) => `vec4<u32>(${a}<${b})`}),
      createOpProgramShader, undefined, undefined, DataType.bool));
};
