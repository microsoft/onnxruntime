import {ComputeContext, GpuDataType, ProgramInfo, ProgramMetadata} from "../types";
import {TensorView} from "../../tensor";
import {DataType} from "../../../wasm-common";
import {ShapeUtil} from "../../util";
import {ShaderHelper} from "./common";
import {AttributeWithCacheKey, createAttributeWithCacheKey} from "../attribute-with-cache-key";

export interface GatherAttributes extends AttributeWithCacheKey {
    axis: number;
}

const validateInputs = (inputs: readonly TensorView[]): void => {
    if (!inputs || inputs.length !== 2) {
        throw new Error('Gather requires 2 inputs.');
    }

    // if (inputs[0].dataType !== DataType.float || inputs[1].dataType !== DataType.float) {
    //     throw new Error('inputs should be float type');
    // }
};

const createGatherProgramInfo =
    (metadata: ProgramMetadata, inputs: readonly TensorView[], attributes: GatherAttributes):
        ProgramInfo => {
        const inputShape = inputs[0].dims;
        let indicesShape = inputs[1].dims;
        if (indicesShape.length === 0) {
            indicesShape = [1];
        }

        const inputRank = indicesShape.length;
        const axis = ShapeUtil.normalizeAxis(attributes.axis, inputRank);

        const outputShape = [];
        for (let i = 0; i < axis; ++i) {
            outputShape.push(inputShape[i]);
        }

        for (const i of indicesShape) {
            outputShape.push(i);
        }

        for (let i = axis + 1; i < inputRank; ++i) {
            outputShape.push(inputShape[i]);
        }
        const inputDataType = inputs[0].dataType;
        const block = ShapeUtil.sizeFromDimension(inputShape, axis + 1);
        const elementSize = [DataType.int64, DataType.uint64].includes(inputDataType) ? 2 : 1;
        const indicesElementSize = [DataType.int64, DataType.uint64].includes(inputs[1].dataType) ? 2 : 1;
        const blockSize = elementSize * block; // TODO: i64
        const M = ShapeUtil.sizeToDimension(inputShape, axis);
        const N = indicesShape.length;
        const dataBatchElements = ShapeUtil.sizeFromDimension(inputShape, axis) * elementSize;
        const gatheredBatchElements = N * block * elementSize;
        const axisDimLimit = inputShape[axis];

        const inputSize = ShapeUtil.size(inputShape);
        const outputSize = ShapeUtil.size(outputShape) || elementSize;

        const totalGathers = M * N;
        console.log('gather!', inputs, outputShape, axis, attributes, totalGathers);
        const dataType = 'i32'; // lets treat everything as i32
        // we treat int64 indices as little endian i32 as you cannot create more than 2gb buffer anyway
        const getShaderSource = (shaderHelper: ShaderHelper) => `
  const N: u32 = ${N};
  const elementSize: i32 = ${elementSize};
  const indicesElementSize: i32 = ${indicesElementSize};

  @group(0) @binding(0) var<storage, read> input : array<${dataType}>;
  @group(0) @binding(1) var<storage, read> inputIndices : array<${dataType}>;
  @group(0) @binding(2) var<storage, read_write> output: array<${dataType}>;

  ${shaderHelper.mainStart()}
    let batch = i32(global_idx / N);
    let i = i32(global_idx % N);

    let srcOffsetBatch: i32 = batch * ${dataBatchElements};
    let dstOffsetBatch: i32 = batch * ${gatheredBatchElements};
    var idx = inputIndices[i * indicesElementSize];
    if (idx < 0) {
        idx = idx + ${axisDimLimit};
    }
    let srcOffset = srcOffsetBatch + idx * ${blockSize};
    let dstOffset = dstOffsetBatch + i * ${blockSize};
    if (srcOffset >= ${inputSize}) {
        return;
    }
    if (dstOffset >= ${outputSize}) {
        return;
    }
    output[dstOffset] = input[srcOffset];
    if (elementSize > 1) {
        output[dstOffset + 1] = input[srcOffset + 1];
    }
  }`;
        return {
            ...metadata,
            outputs: [
                {dims: outputShape, dataType: inputs[0].dataType, gpuDataType: GpuDataType.default},
                // {dims: meanInvStdDevDim, dataType: inputs[0].dataType, gpuDataType: GpuDataType.default},
                // {dims: meanInvStdDevDim, dataType: inputs[0].dataType, gpuDataType: GpuDataType.default},
            ],
            getShaderSource,
            dispatchGroup: () => ({x: Math.ceil( totalGathers / 64 /* workgroup size */) || 1})
        };
    };

export const parseGatherAttributes = (attributes: Record<string, unknown>): GatherAttributes =>
    createAttributeWithCacheKey(attributes as Omit<GatherAttributes, keyof AttributeWithCacheKey>);

export const gather = (context: ComputeContext, attributes: GatherAttributes): void => {
    validateInputs(context.inputs);

    const metadata = {
        name: 'Gather',
        inputTypes: [GpuDataType.default, GpuDataType.default],
        cacheHint: attributes.cacheKey,
    };

    context.compute(createGatherProgramInfo(metadata, context.inputs, attributes));
};
