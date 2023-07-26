// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {ComputeContext, GpuDataType, ProgramInfo, ProgramMetadata} from '../types';
import {TensorView} from '../../tensor';
import {DataType, tensorTypeToWsglType} from '../../../wasm-common';
import {ShapeUtil} from '../../util';
import {getMaxWorkgroupLimits, ShaderHelper} from './common';
import {AttributeWithCacheKey, createAttributeWithCacheKey} from '../attribute-with-cache-key';

export interface LayerNormAttributes extends AttributeWithCacheKey {
    axis: number;
    epsilon: number;
}

const validateInputs = (inputs: readonly TensorView[]): void => {
    if (!inputs || inputs.length !== 3) {
        throw new Error('layerNorm requires 3 inputs.');
    }

    if (inputs[0].dataType !== DataType.float || inputs[1].dataType !== DataType.float) {
        throw new Error('inputs should be float type');
    }
};

const createLayerNormProgramInfo =
    (metadata: ProgramMetadata, inputs: readonly TensorView[], attributes: LayerNormAttributes):
        ProgramInfo => {
        const xShape = inputs[0].dims;
        const scale = inputs[1];
        const bias = inputs[2];

        const outputShape = xShape;
        const outputSize = ShapeUtil.size(outputShape);
        const axis = ShapeUtil.normalizeAxis(attributes.axis, xShape.length);
        const normCount = ShapeUtil.sizeToDimension(xShape, axis);
        const normSize = ShapeUtil.sizeFromDimension(xShape, axis);

        const scaleSize = ShapeUtil.size(scale.dims);
        const biasSize = bias ? ShapeUtil.size(bias.dims) : 0;
        if (scaleSize !== normSize || (bias && biasSize !== normSize)) {
            throw new Error(`Size of X.shape()[axis:] == ${normSize}.
             Size of scale and bias (if provided) must match this. 
             Got scale size of ${scaleSize} and bias size of ${biasSize}`);
        }

        const meanInvStdDevDim = [];
        for (let i = 0; i < xShape.length; ++i) {
            if (i < axis) {
                meanInvStdDevDim.push(xShape[i]);
            } else {
                meanInvStdDevDim.push(1);
            }
        }

        const dataType = tensorTypeToWsglType(inputs[0].dataType);
        const [dispatchGroup, workgroupLimits] = getMaxWorkgroupLimits(normCount);

        const getShaderSource = (shaderHelper: ShaderHelper) => `
  const normSize: u32 = ${normSize};
  const normSizeTyped: ${dataType} = ${normSize};
  const epsilon: f32 = ${attributes.epsilon};

  @group(0) @binding(0) var<storage, read> x : array<${dataType}>;
  @group(0) @binding(1) var<storage, read> scale : array<${dataType}>;
  @group(0) @binding(2) var<storage, read> bias : array<${dataType}>;
  @group(0) @binding(3) var<storage, read_write> output : array<${dataType}>;
  @group(0) @binding(4) var<storage, read_write> meanDataOutput : array<${dataType}>;
  // @group(0) @binding(5) var<storage, read_write> invStdOutput : array<${dataType}>;

  ${shaderHelper.mainStart(workgroupLimits)}
    let offset = global_idx * normSize;
    if (offset >= ${outputSize}) { return; }
    var mean: ${dataType} = 0;
    var meanSquare: ${dataType} = 0;

    for (var h: u32 = 0u; h < normSize; h++) {
        mean = mean + x[h + offset];
        meanSquare = meanSquare + x[h + offset] * x[h + offset];
    }
    mean = mean / normSizeTyped;
    meanSquare = sqrt(meanSquare / normSizeTyped - mean * mean + epsilon);

    for (var j: u32 = 0; j < normSize; j++) {
        output[j + offset] = (x[j + offset] - mean) / meanSquare * scale[j] + bias[j];
    }

    meanDataOutput[global_idx] = mean;
    //invStdOutput[global_idx] = 1 / meanSquare;
  }`;
        return {
            ...metadata,
            outputs: [
                {dims: outputShape, dataType: inputs[0].dataType, gpuDataType: GpuDataType.default},
                {dims: meanInvStdDevDim, dataType: inputs[0].dataType, gpuDataType: GpuDataType.default},
                // {dims: meanInvStdDevDim, dataType: inputs[0].dataType, gpuDataType: GpuDataType.default},
            ],
            getShaderSource,
            dispatchGroup: () => (dispatchGroup)
        };
    };

export const parseLayerNormAttributes = (attributes: LayerNormAttributes): LayerNormAttributes =>
    createAttributeWithCacheKey({ axis: attributes.axis, epsilon: attributes.epsilon });

export const layerNorm = (context: ComputeContext, attributes: LayerNormAttributes): void => {
    validateInputs(context.inputs);

    const metadata = {
        name: 'LayerNorm',
        inputTypes: [GpuDataType.default, GpuDataType.default, GpuDataType.default],
        cacheHint: attributes.cacheKey,
    };

    context.compute(createLayerNormProgramInfo(metadata, context.inputs, attributes));
};
