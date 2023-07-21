import {ComputeContext, GpuDataType, ProgramInfo, ProgramMetadata} from "../types";
import {TensorView} from "../../tensor";
import {DataType} from "../../../wasm-common";
import {ShapeUtil} from "../../util";
import {ShaderHelper} from "./common";
import {AttributeWithCacheKey, createAttributeWithCacheKey} from "../attribute-with-cache-key";

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

        const dataType = 'f32';
        const getShaderSource = (shaderHelper: ShaderHelper) => `
  const normSize: u32 = ${normSize}u;
  const epsilon: f32 = ${attributes.epsilon};

  @group(0) @binding(0) var<storage, read> x : array<${dataType}>;
  @group(0) @binding(1) var<storage, read> scale : array<${dataType}>;
  @group(0) @binding(2) var<storage, read> bias : array<${dataType}>;
  @group(0) @binding(3) var<storage, read_write> output : array<${dataType}, ${outputSize}>;
  @group(0) @binding(3) var<storage, read_write> meanDataOutput : array<${dataType}, ${normCount}>;
  @group(0) @binding(3) var<storage, read_write> invStdOutput : array<${dataType}, ${normCount}>;

  ${shaderHelper.mainStart(normCount)}
    ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes(outputSize)}
    let offset = global_idx * normSize;
    let mean = 0u;
    let meanSquare = 0u;

    for (var h: u32: 0u; h<normSize; h++) {
        mean = mean + x[h + offset];
        meanSquare = x[h + offst] * x[h + offset];
    }
    mean = mean / normSize;
    meanSquare = sqrt(meanSquare / normSize - mean * mean + epsilon)
    
    for (var h: u32: 0u; h<normSize; h++) {
        output[h + offset] = (x[h + offset] - mean) / mean_square * scale[h] + bias[h];
    }
    
    meanDataOutput[global_idx] = mean
    invStdOutput[global_idx] = 1 / meanSquare
  }`;
        return {
            ...metadata,
            outputs: [
                {dims: outputShape, dataType: inputs[0].dataType, gpuDataType: GpuDataType.default},
                {dims: meanInvStdDevDim, dataType: inputs[0].dataType, gpuDataType: GpuDataType.default},
                {dims: meanInvStdDevDim, dataType: inputs[0].dataType, gpuDataType: GpuDataType.default},
            ],
            getShaderSource,
            dispatchGroup: () => ({x: Math.ceil(outputSize / 64 /* workgroup size */)})
        };
    };

export const parseLayerNormAttributes = (attributes: Record<string, unknown>): LayerNormAttributes =>
    createAttributeWithCacheKey(attributes as Omit<LayerNormAttributes, keyof AttributeWithCacheKey>);

export const layerNorm = (context: ComputeContext, attributes: LayerNormAttributes): void => {
    validateInputs(context.inputs);

    const metadata = {
        name: 'LayerNorm',
        inputTypes: [GpuDataType.default, GpuDataType.default, GpuDataType.default],
        cacheHint: attributes.cacheKey,
    };

    context.compute(createLayerNormProgramInfo(metadata, context.inputs, attributes));
};
