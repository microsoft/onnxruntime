// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {DataType} from '../../../wasm-common';
import {TensorView} from '../../tensor';
import {BroadcastUtil, ShapeUtil} from '../../util';
import {GpuDataType, ProgramInfo, ProgramInfoLoader, ProgramMetadata} from '../types';

import {IndicesHelper, ShaderHelper} from './common';

type BuiltinFunctionName = string;
export type BinaryCustomExpression = (expressionA: string, expressionB: string, expressionC?: string) => string;
export type BinaryFunctionCall = BuiltinFunctionName|BinaryCustomExpression|{
  scalar: BinaryCustomExpression;
  vector: BinaryCustomExpression;
};

type CreateOpProgramShader =
    (shaderHelper: ShaderHelper, inputs: readonly TensorView[], dimsOutput: readonly number[], vectorize: boolean,
     doBroadcast: boolean, funcCall: BinaryFunctionCall, typeOutput: number, additionalImplementation?: string) =>
        string;

/* eslint-disable no-param-reassign */
const createOpProgramInfo =
    (metadata: ProgramMetadata, inputs: readonly TensorView[], funcCall: BinaryFunctionCall,
     createOpProgramShader: CreateOpProgramShader, additionalImplementation?: string,
     outputDataType?: number): ProgramInfo => {
      const a = inputs.length === 3 ? inputs[1] : inputs[0];
      const b = inputs.length === 3 ? inputs[2] : inputs[1];
      if (outputDataType == null) {
        outputDataType = inputs.length === 3 ? inputs[1].dataType : inputs[0].dataType;
      }

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
        for (let i = 1; i < outputShape.length; i++) {
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
        getShaderSource: (shaderHelper) => createOpProgramShader(
            shaderHelper, inputs, outputShape, vectorize, isBroadcast, funcCall, outputDataType as number,
            additionalImplementation),
        outputs: [{dims: outputShape, dataType: outputDataType, gpuDataType: GpuDataType.default}],
        dispatchGroup: () =>
            ({x: Math.ceil(outputSize / 64 /* workgroup size */ / (vectorize ? 4 : 1) /* vec size */)})
      };
    };

// This is used for ops like binary, where.
export const createOpProgramInfoLoader =
    (inputs: readonly TensorView[], name: string, funcCall: BinaryFunctionCall,
     createOpProgramShader: CreateOpProgramShader, additionalImplementation?: string, cacheKey?: string,
     outputDataType?: number): ProgramInfoLoader => {
      const inputTypes = inputs.length === 3 ? [GpuDataType.default, GpuDataType.default, GpuDataType.default] :
                                               [GpuDataType.default, GpuDataType.default];
      const metadata: ProgramMetadata = {name, inputTypes, cacheHint: cacheKey};
      return {
        ...metadata,
        get: () => createOpProgramInfo(
            metadata, inputs, funcCall, createOpProgramShader, additionalImplementation, outputDataType)
      };
    };

export const calcOffsetImpl = (
    name: string,
    dims: readonly number[],
    output: IndicesHelper,
    dimsOutput: readonly number[],
    ) => {
  const strides = ShapeUtil.computeStrides(dims);
  const offsets: string[] = [];
  for (let i = dims.length - 1; i >= 0; i--) {
    const idx = output.indicesGet('outputIndices', i + dimsOutput.length - dims.length);
    offsets.push(`${strides[i]}u * (${idx} % ${dims[i]}u)`);
  }
  return `fn calcOffset${name}(outputIndices: ${output.type.indices}) -> u32 {
           return ${offsets.length > 0 ? offsets.join('+') : '0u'};
         }
        `;
};

export const getIndexComponent = (name: string, x: number) => (`
      let offset${name}${x} = calcOffset${name}(outputIndices${x});
      let index${name}${x} = offset${name}${x} / 4u;
      let component${name}${x} = offset${name}${x} % 4u;
    `);

type SingleAssignmentFuncCall = (resStr: string, x: number, typeCast?: string) => string;
export const fourAssignment = (singleAssignment: SingleAssignmentFuncCall, typeOutput: number) => {
  let assignment = '';
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
  return assignment;
};
