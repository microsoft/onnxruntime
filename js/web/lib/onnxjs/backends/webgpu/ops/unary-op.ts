// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {Tensor} from '../../../tensor';
import {WebGpuInferenceHandler} from '../inference-handler';
import {GpuDataType} from '../types';

export const abs = (handler: WebGpuInferenceHandler, inputs: Tensor[]): Tensor[] => handler.run(
    {
      name: 'Abs',
      inputTypes: [GpuDataType.default],
      // inputLayouts: [],
      // outputLayouts: [],
      shaderSource: `
      @group(0) @binding(0) var<storage, read> inputData : array<f32>;
      @group(0) @binding(1) var<storage, write> outputData : array<f32>;

      @stage(compute) @workgroup_size(32)
      fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
        // Guard against out-of-bounds work group sizes
        if (global_id.x * 32u >= ${inputs[0].size}u) {
          return;
        }

        //
        // TODO: SIMD?
        //

        let start = global_id.x * 32u;
        let end = select(start + 32u, ${inputs[0].size}u, start + 32u > ${inputs[0].size}u);

        for (var i = start; i < end; i = i + 1u) {
          outputData[i] = abs(inputData[i]);
        }
      }`,
      outputs: [{dims: inputs[0].dims, type: inputs[0].type, gpuDataType: GpuDataType.default}],
      // entryPoint: 'main',
      dispatchGroup: (inputTensors) => ({x: Math.ceil(inputTensors[0].size / 32)})
    },
    inputs);
