/*
 * // Copyright (c) Microsoft Corporation. All rights reserved.
 * // Licensed under the MIT License.
 */

import {DataType} from '../../../wasm-common';
import {TensorView} from '../../tensor';
import {AttributeWithCacheKey, createAttributeWithCacheKey} from '../attribute-with-cache-key';
import {ComputeContext, GpuDataType, ProgramInfo, ProgramInfoLoader, ProgramMetadata} from '../types';
import {ShapeUtil} from "../../util";
import {createIndicesHelper, ShaderHelper} from "./common";
import {ReduceAttributes} from "./reduce";
import {GemmAttributes} from "./gemm";

export interface GatherAttributes extends AttributeWithCacheKey {
    readonly axis: number;
}

const validateInputsAndAttribute = (inputs: readonly TensorView[], attribute: GatherAttributes): void => {
    const data = inputs[0];
    const indices = inputs[1];
    const rankOfData = data.dims.length;
    if (!inputs || inputs.length !== 2) {
        throw new Error('Gather op requires 2 inputs.');
    }

    if (!attribute) {
        throw new Error('Invalid Gather attribute. There is not attribute provided.');
    }

    if (data.dims.length < 1) {
        throw new Error('Invalid Gather input data shape.');
    }

    if (attribute.axis < -rankOfData || attribute.axis >= rankOfData) {
        throw new
        Error(`Invalid Gather attribute axis. The acceptable range for axis is [${-rankOfData}, ${rankOfData - 1}]`);
    }

    if (indices.dataType !== DataType.int64 && indices.dataType !== DataType.int32) {
        throw new Error('Invalid Gather input data input type. Accepted input data type is int32 or int64.');
    }
};

type GatherOp = (inputs: readonly TensorView[], axis: number) => string[];

const createGatherProgramInfo =
    (metadata: ProgramMetadata, inputs: readonly TensorView[], attributes: GatherAttributes,
     gatherOp: GatherOp): ProgramInfo => {
        const outputShape: number[] = [];
        const inputShape = inputs[0].dims;

        const idxCopy: string[] = [];  // copy output indexes to input indexes

        const axis = attributes.axis;
        const ops = gatherOp(inputs, axis);
        const inputIndicesHelper = createIndicesHelper('input', inputShape);
        const initInputIdx =
            (ops[1] === '') ? '' : `let inputIdx = ${inputIndicesHelper.i2oExpression('inputIndices')};`;
        let gatherOps = `
          let inputIdx = ${inputIndicesHelper.i2oExpression('inputIndices')};
          ${ops[2]};`;

        for (let k = 0; k < inputs[0].dims.length; k++) {
            // if this axis is reduced
            gatherOps = `for(var j${k}: u32 = 0; j${k} < ${inputs[0].dims[k]}; j${k}++) {
                            inputIndices[${k}] = j${k};
                            ${gatherOps}
                          }`;
        }

        const outputIndicesHelper = createIndicesHelper('output', outputShape);
        const outputSize = ShapeUtil.size(outputShape);
        const dataType = 'f32';

        const getShaderSource = (shaderHelper: ShaderHelper) => `
          @group(0) @binding(0) var<storage, read> _A : array<${dataType}>;
          @group(0) @binding(1) var<storage, read_write> output : array<${dataType}>;

          ${outputIndicesHelper.o2iImpl}
          ${inputIndicesHelper.i2oImpl}

          ${shaderHelper.mainStart()}
          ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes(outputSize)}
          ${inputIndicesHelper.indicesVariableDeclaration('inputIndices')}
          ${outputIndicesHelper.indicesVariableDeclaration('outputIndices')}
          ${outputIndicesHelper.o2iCall('global_idx', 'outputIndices')}

          var value = ${dataType}(0);

          ${idxCopy.join('\n')}
          ${ops[0]}       // init ops for reduce max/min
          ${initInputIdx}
          ${ops[1]}
          ${gatherOps}
          ${ops[3]}       // final computation for reduce mean
          output[global_idx] = value;
        }`;

        return {
            ...metadata,
            getShaderSource,
            outputs: [{dims: outputShape, dataType: inputs[0].dataType, gpuDataType: GpuDataType.default}],
            dispatchGroup: () => ({x: Math.ceil(outputSize / 64 /* workgroup size */)})
        };
    };
const createGatherAttributesFromInput = (input: TensorView, attributes: GatherAttributes): GatherAttributes => {
    const axis: number = attributes.axis;
    return createAttributeWithCacheKey({axis});
};
const createGatherProgramInfoLoader =
    (inputs: readonly TensorView[], name: string, attributes: GatherAttributes, gatherOp: GatherOp):
        ProgramInfoLoader => {
        const metadata: ProgramMetadata = {name, inputTypes: [GpuDataType.default]};
        return {
            ...metadata,
            get: () => createGatherProgramInfo(
                metadata, inputs,
                attributes,
                gatherOp
            )
        };
    };
export const gather = (context: ComputeContext, attribute: GatherAttributes): void => {
    validateInputsAndAttribute(context.inputs, attribute);
    const gatherOp: GatherOp = (): string[] => ['value = 1.0;', '', 'value *= _A[inputIdx];', ''];
    context.compute(createGatherProgramInfoLoader(context.inputs, 'Gather', attribute, gatherOp));
};
export const parseGatherAttributes = (attributes: Record<string, unknown>): GatherAttributes =>
    createAttributeWithCacheKey(attributes as Omit<GatherAttributes, keyof AttributeWithCacheKey>);
