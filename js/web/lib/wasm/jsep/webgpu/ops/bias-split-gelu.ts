// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {DataType, tensorTypeToWsglType} from '../../../wasm-common';
import {TensorView} from '../../tensor';
import {ComputeContext, GpuDataType, ProgramInfo, ProgramMetadata} from '../types';
import {ShaderHelper} from './common';
import {erfImpl} from './unary-op';

const validateInputs = (inputs: readonly TensorView[]): void => {
    if (inputs[0].dataType !== DataType.float) {
        throw new Error('inputs should be float type');
    }

    if (inputs[0].dims.length !== 3) {
        throw new Error('input should have 3 dimensions');
    }

    if (![2560, 5120, 10240].includes(inputs[0].dims[2])) {
        throw new Error('hidden state should be 2560, 5120 or 10240');
    }

    if (inputs[1].dims.length !== 1) {
        throw new Error('bias is expected to have 1 dimensions');
    }

    if (inputs[0].dims[2] !== inputs[1].dims[0]) {
        throw new Error('last dimension of input and bias are not the same');
    }
};

const createBiasSplitGeluProgramInfo =
    (metadata: ProgramMetadata, inputs: readonly TensorView[]):
        ProgramInfo => {
        const input = inputs[0];
        const outputShape = input.dims.slice();
        outputShape[2] = outputShape[2] / 2;

        const gridSize = outputShape[0] * outputShape[1];
        const halfHiddenSize = outputShape[2];
        const dataType = tensorTypeToWsglType(inputs[0].dataType);
        const blockSize = halfHiddenSize / 256;
        const outputSize = gridSize;

        const getShaderSource = (shaderHelper: ShaderHelper) => `
  const M_SQRT2 = sqrt(2.0);
  const TPB = 256;

  @group(0) @binding(0) var<storage, read> input : array<${dataType}>;
  @group(0) @binding(1) var<storage, read> bias : array<${dataType}>;
  @group(0) @binding(2) var<storage, read_write> output : array<${dataType}>;

  ${erfImpl('f32')}

  ${shaderHelper.mainStart()}
    if (global_idx >= ${outputSize}) {
        return;
    }
    let blockIdx = global_idx / ${halfHiddenSize};
    let threadIdx = global_idx % ${halfHiddenSize};
    
    var indexInput = blockIdx * ${halfHiddenSize} * 2 + threadIdx;
    var indexOutput = blockIdx * ${halfHiddenSize} + threadIdx;
    var indexBias = threadIdx;

    for (var h: u32 = 0u; h < ${blockSize}; h++) {
      let valueLeft = input[indexInput] + bias[indexBias];
      let valueRight = input[indexInput + ${halfHiddenSize}] + bias[indexBias + ${halfHiddenSize}];
      
      let geluRight = valueRight * 0.5 * (erf_vf32(valueRight / M_SQRT2) + 1);
      output[indexOutput] = valueLeft * geluRight;
      indexInput += TPB;
      indexOutput += TPB;
      indexBias += TPB;
    }
  }`;

        return {
            ...metadata,
            outputs: [{dims: outputShape, dataType: inputs[0].dataType, gpuDataType: GpuDataType.default}],
            getShaderSource,
            dispatchGroup: () => ({x: Math.ceil(outputSize / 64 /* workgroup size */)})
        };
    };

export const biasSplitGelu = (context: ComputeContext): void => {
    validateInputs(context.inputs);

    const metadata = {
        name: 'BiasSplitGelu',
        inputTypes: [GpuDataType.default, GpuDataType.default],
    };

    context.compute(createBiasSplitGeluProgramInfo(metadata, context.inputs));
};
