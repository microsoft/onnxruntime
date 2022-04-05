// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {AttributeWithCacheKey, createAttributeWithCacheKey} from '../../../attribute-with-cache-key';
import {Tensor} from '../../../tensor';
import {BroadcastUtil, ShapeUtil} from '../../../util';
import {WebGpuInferenceHandler} from '../inference-handler';
import {GpuDataType, ProgramInfo, ProgramInfoLoader, ProgramMetadata} from '../types';
import {WORKGROUP_SIZE} from './common';

type BinaryFunctionImplementation =
  // name, builtin function call.
  // eg. ['Pow', 'pow']
  [string, string]|
  // name, function call builder, extra implementation (optional)
  // eg. ['Add', (a, b) => `${a}+${b}`]
  [string, (variableNameA: string, variableNameB: string) => string, string?];

const createBinaryOpProgramShader =
    (functionImplementation: BinaryFunctionImplementation, vectorize: boolean, doBroadcast: boolean,
     dimsA: readonly number[], dimsB: readonly number[], dimsOutput: readonly number[]) => {
      const outputSize = ShapeUtil.size(dimsOutput);
      const vecSize = Math.ceil(outputSize / 4);
      return `
  let WORKGROUP_SIZE: u32 = ${WORKGROUP_SIZE}u;

  @group(0) @binding(0) var<storage, read> inputData : array<vec4<f32>>;
  @group(0) @binding(1) var<storage, write> outputData : array<vec4<f32>>;

  ${funcImpl}

  @stage(compute) @workgroup_size(WORKGROUP_SIZE)
  fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {

    // Guard against out-of-bounds work group sizes
    if (global_id.x >= ${vecSize}u) {
      return;
    }

    outputData[global_id.x] = ${funcName}(inputData[global_id.x]);
  }`;
    };

const createBinaryOpProgramInfo =
    (metadata: ProgramMetadata, a: Tensor, b: Tensor, functionImplementation: BinaryFunctionImplementation,
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
        if (a.dims.length > 0 && b.dims.length > 0) {
          const lastNotOneDimensionA
          vectorize = false;
        }


      } else {
        // element-wise
        vectorize = true;
      }

      return {
        ...metadata,
        shaderSource: createBinaryOpProgramShader(functionImplementation, vectorize, a.dims, b.dims, outputShape),
        outputs: [{dims: outputShape, type: outputTensorType, gpuDataType: GpuDataType.default}],
        dispatchGroup: () =>
            ({x: Math.ceil(outputSize / 64 /* workgroup size */ / (vectorize ? 4 : 1) /* vec size */)})
      };
    };

const createBinaryOpProgramInfoLoader =
    (inputs: Tensor[], functionImplementation: BinaryFunctionImplementation, cacheKey?: string): ProgramInfoLoader => {
      const metadata: ProgramMetadata = {
        name: functionImplementation[0],
        inputTypes: [GpuDataType.default, GpuDataType.default],
        cacheHint: cacheKey
      };
      return {
        ...metadata,
        get: () => createBinaryOpProgramInfo(metadata, inputs[0], inputs[1], functionImplementation)
      };
    };

export const add = async(handler: WebGpuInferenceHandler, inputs: Tensor[]): Promise<Tensor[]> =>
    handler.run(createBinaryOpProgramInfoLoader(inputs, ['Add', (a, b) => `${a}+${b}`]), inputs);
