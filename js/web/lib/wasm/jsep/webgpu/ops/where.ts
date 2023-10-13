// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {DataType} from '../../../wasm-common';
import {TensorView} from '../../tensor-view';
import {BroadcastUtil, ShapeUtil} from '../../util';
import {ComputeContext, ProgramInfo} from '../types';

import {inputVariable, outputVariable, ShaderHelper} from './common';

const createWhereOpProgramShader =
    (shaderHelper: ShaderHelper, inputs: readonly TensorView[], dimsOutput: readonly number[], isBroadcast: boolean,
     typeOutput: number) => {
      const outputSize = ShapeUtil.size(dimsOutput);
      const vecSize = Math.ceil(outputSize / 4);

      const output = outputVariable('outputData', typeOutput, dimsOutput, 4);
      const a = inputVariable('aData', inputs[1].dataType, inputs[1].dims, 4);
      const b = inputVariable('bData', inputs[2].dataType, inputs[2].dims, 4);
      const c = inputVariable('cData', inputs[0].dataType, inputs[0].dims, 4);

      let assignment: string;
      const expression = (a: string, b: string, c: string) => `select(${b}, ${a}, ${c})`;
      if (!isBroadcast) {
        assignment = output.setByOffset(
            'global_idx',
            expression(a.getByOffset('global_idx'), b.getByOffset('global_idx'), c.getByOffset('global_idx')));
      } else {
        const singleAssignment = (resStr: string, x: number, typeCast = '') => {
          const expressionA = `aData[indexA${x}][componentA${x}]`;
          const expressionB = `bData[indexB${x}][componentB${x}]`;
          // eslint-disable-next-line no-bitwise
          const expressionC = `bool(cData[indexC${x}] & ${0xff000000 >>> ((3 - x) * 8)}u)`;
          return `
            let outputIndices${x} = ${output.offsetToIndices(`global_idx * 4u + ${x}u`)};
            let offsetA${x} = ${a.broadcastedIndicesToOffset(`outputIndices${x}`, output)};
            let offsetB${x} = ${b.broadcastedIndicesToOffset(`outputIndices${x}`, output)};
            let offsetC${x} = ${c.broadcastedIndicesToOffset(`outputIndices${x}`, output)};
            let indexA${x} = offsetA${x} / 4u;
            let indexB${x} = offsetB${x} / 4u;
            let indexC${x} = offsetC${x} / 4u;
            let componentA${x} = offsetA${x} % 4u;
            let componentB${x} = offsetB${x} % 4u;
            ${resStr}[${x}] = ${typeCast}(${expression(expressionA, expressionB, expressionC)});
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
        ${shaderHelper.declareVariables(c, a, b, output)}
        ${shaderHelper.mainStart()}
        ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes(vecSize)}
        ${assignment}
      }`;
    };

const createWhereOpProgramInfo = (inputs: readonly TensorView[]): ProgramInfo => {
  const dimsA = inputs[1].dims;
  const dimsB = inputs[2].dims;
  const dimsC = inputs[0].dims;
  const outputDataType = inputs[1].dataType;

  const isBroadcast = !(ShapeUtil.areEqual(dimsA, dimsB) && ShapeUtil.areEqual(dimsB, dimsC));
  let outputShape = dimsA;
  let outputSize = ShapeUtil.size(dimsA);
  // TODO: deal with zero-sized tensors (eg. dims=[1,0])

  if (isBroadcast) {
    const calculatedShape = BroadcastUtil.calcShape(BroadcastUtil.calcShape(dimsA, dimsB, false)!, dimsC, false);
    if (!calculatedShape) {
      throw new Error('Can\'t perform where op on the given tensors');
    }
    outputShape = calculatedShape;
    outputSize = ShapeUtil.size(outputShape);
  }

  return {
    name: 'Where',
    getShaderSource: (shaderHelper) =>
        createWhereOpProgramShader(shaderHelper, inputs, outputShape, isBroadcast, outputDataType),
    getRunData: () => ({
      outputs: [{dims: outputShape, dataType: outputDataType}],
      dispatchGroup: {x: Math.ceil(outputSize / 64 /* workgroup size */ / 4 /* vec size */)}
    }),
  };
};

export const where = (context: ComputeContext): void => {
  context.compute(createWhereOpProgramInfo(context.inputs));
};
