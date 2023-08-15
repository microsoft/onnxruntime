// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {TensorView} from '../../tensor';
import {ShapeUtil} from '../../util';
import {ComputeContext} from '../types';

import {BinaryCustomExpression, BinaryFunctionCall, calcOffsetImpl, createOpProgramInfoLoader, fourAssignment, getIndexComponent} from './binary-like-util';
import {inputVariable, outputVariable, ShaderHelper} from './common';

const createOpProgramShader =
    (shaderHelper: ShaderHelper, inputs: readonly TensorView[], dimsOutput: readonly number[], vectorize: boolean,
     doBroadcast: boolean, funcCall: BinaryFunctionCall, typeOutput: number, additionalImplementation?: string) => {
      const typeA = inputs[1].dataType;
      const typeB = inputs[2].dataType;
      const typeC = inputs[0].dataType;
      const dimsA = inputs[1].dims;
      const dimsB = inputs[2].dims;
      const dimsC = inputs[0].dims;
      const outputSize = ShapeUtil.size(dimsOutput);
      const vecSize = Math.ceil(outputSize / 4);

      let expressionScalar: BinaryCustomExpression;
      let expressionVector: BinaryCustomExpression;
      if (typeof funcCall === 'string') {
        expressionScalar = expressionVector = (a, b, c) => `${funcCall}((${a}),(${b}),(${c}))`;
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
      const c = inputVariable('cData', typeC, dimsC, 4);
      if (doBroadcast) {
        broadcastImpl = `
          ${calcOffsetImpl('A', dimsA, output, dimsOutput)}
          ${calcOffsetImpl('B', dimsB, output, dimsOutput)}
          ${calcOffsetImpl('C', dimsC, output, dimsOutput)}
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
                  'global_idx',
                  expressionVector(
                      a.getByOffset('offsetA / 4u'), b.getByOffset('offsetB / 4u'), c.getByOffset('offsetC / 4u')))}`;
        } else {
          assignment = output.setByOffset(
              'global_idx',
              expressionVector(a.getByOffset('global_idx'), b.getByOffset('global_idx'), c.getByOffset('global_idx')));
        }
      } else {
        if (!doBroadcast) {
          throw new Error('no necessary to use scalar implementation for element-wise binary op implementation.');
        }

        const singleAssignment = (resStr: string, x: number, typeCast = '') => {
          const expressionA = `aData[indexA${x}][componentA${x}]`;
          const expressionB = `bData[indexB${x}][componentB${x}]`;
          // eslint-disable-next-line no-bitwise
          const expressionC = `bool(cData[indexC${x}] & ${0xff000000 >>> ((3 - x) * 8)}u)`;
          return `
            let outputIndices${x} = ${output.offsetToIndices(`global_idx * 4u + ${x}u`)};
            ${getIndexComponent('A', x)}
            ${getIndexComponent('B', x)}
            ${getIndexComponent('C', x)}
            ${resStr}[${x}] = ${typeCast}(${expressionScalar(expressionA, expressionB, expressionC)});
          `;
        };
        assignment = fourAssignment(singleAssignment, typeOutput);
      }

      return `
        ${shaderHelper.declareVariables(c, a, b, output)}

        ${additionalImplementation ?? ''}
        ${broadcastImpl}

        ${shaderHelper.mainStart()}
        ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes(vecSize)}
        ${assignment}
      }`;
    };

export const where = (context: ComputeContext): void => {
  const type = inputVariable('input', context.inputs[1].dataType, context.inputs[1].dims).type.value;
  context.compute(
      createOpProgramInfoLoader(
          context.inputs, 'Where', ({
            scalar: (a, b, c) => `where_custom(${a}, ${b}, ${c})`,
            vector: (a, b, c) => `where_vector_custom(${a}, ${b}, ${c})`
          }),
          createOpProgramShader, `
  fn where_custom(a : ${type}, b : ${type}, cond : bool) -> ${type} {
    if cond == true {
      return a;
    }
    return b;
  }

  fn where_vector_custom(x : vec4<${type}>, y : vec4<${type}>, cond : vec4<bool>) -> vec4<${type}> {
    return vec4<${type}>(where_custom(x.x, y.x, cond.x), where_custom(x.y, y.y, cond.y),
             where_custom(x.z, y.z, cond.z), where_custom(x.w, y.w, cond.w));
   }
   `),
  );
};
