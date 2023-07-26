// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.


import {DataType} from '../../../wasm-common';
import {TensorView} from '../../tensor';
import {AttributeWithCacheKey, createAttributeWithCacheKey} from '../attribute-with-cache-key';
import {ComputeContext, GpuDataType, ProgramInfo} from '../types';
import {ShapeUtil} from '../../util';
import {ShaderHelper} from './common';
// import {createTransposeProgramInfo, TransposeAttributes, transposeProgramMetadata} from "./transpose";
// import {createTransposeProgramInfo, TransposeAttributes, transposeProgramMetadata} from './transpose';

const validateInputs = (inputs: readonly TensorView[]): void => {
	const data = inputs[0];
	const indices = inputs[1];
	if (!inputs || inputs.length !== 2) {
		throw new Error('Gather op requires 2 inputs.');
	}

	if (data.dims.length < 1) {
		throw new Error('Invalid Gather input data shape.');
	}

	if (indices.dataType !== DataType.int64 && indices.dataType !== DataType.int32) {
		throw new Error('Invalid Gather input data input type. Accepted input data type is int32 or int64.');
	}
};

export interface GatherAttributes extends AttributeWithCacheKey {
	readonly axis: number;
}

export const gatherProgramMetadata = {
	name: 'Gather',
	inputTypes: [GpuDataType.default,GpuDataType.default],
};
const createGatherProgramInfo = (
	inputData: TensorView,
	index: TensorView
): ProgramInfo => {
	// eslint-disable-next-line no-console
	console.log('createGatherProgramInfo');
	const inputShape = inputData.dims.slice();
	const indexShape = index.dims.slice();
	const outputShape = new Array(inputShape.length + indexShape.length - 1);
	for (let i = 0; i < outputShape.length; i++) {
		if (i < indexShape.length) {  // B
			outputShape[i] = indexShape[i];
		} else {                                                       // C
			outputShape[i] = inputShape[i - indexShape.length + 1];  // skip 1 for axis
		}
	}
	const outputSize = ShapeUtil.size(outputShape);
	const indexSize = ShapeUtil.size(indexShape);
	const dataType = 'f32';  // TODO: support other data type
	const subOutputSize = ShapeUtil.size(outputShape.slice(1));

	// Step 1: transpose input data along axis
	const getShaderSource = (shaderHelper: ShaderHelper) => `
        @group(0) @binding(0) var<storage, read> inputData : array<${dataType}>;
        @group(0) @binding(1) var<storage, read> index : array<i32>;
        @group(0) @binding(2) var<storage, read_write> output : array<${dataType}>;

        ${shaderHelper.mainStart()}
        	${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes(outputSize)}
        	for (var i : i32 = 0; i < ${indexSize}; i++) {
						for (var j : i32 = 0; j < ${subOutputSize}; j++) {
						 let idx = index[i];
<!--						 output[i * ${subOutputSize} + j] = inputData[idx * ${subOutputSize} + j];-->
						}
        	}
    }`;
	return {
		...gatherProgramMetadata,
		getShaderSource,
		outputs: [{dims: outputShape, dataType: inputData.dataType, gpuDataType: GpuDataType.default}],
		dispatchGroup: () => ({x: Math.ceil(outputSize / 64 /* workgroup size */)})
	};
};

export const gather = (context: ComputeContext, attributes: GatherAttributes): void => {
	validateInputs(context.inputs);
	const inputData = context.inputs[0];
	const inputShape = inputData.dims.slice();
	const axis = ShapeUtil.normalizeAxis(attributes.axis, inputShape.length);
	const perm = inputShape;
	perm[axis] = 0;
	perm[0] = axis;
	// const weightTransposeAttribute: TransposeAttributes = createAttributeWithCacheKey({perm});
	// const transposedInput = context.compute({
	// 		...transposeProgramMetadata,
	// 		cacheHint: weightTransposeAttribute.cacheKey,
	// 		get: () => createTransposeProgramInfo(inputData, weightTransposeAttribute.perm)
	// 	}
	// )[0];
	//
	// const gathered = context.compute({
	// 		...gatherProgramMetadata,
	// 		cacheHint: attributes.cacheKey,
	// 		get: () => createGatherProgramInfo(transposedInput, context.inputs[1])
	// 	}
	// )[0];
	// context.compute({
	// 	...transposeProgramMetadata,
	// 	cacheHint: weightTransposeAttribute.cacheKey,
	// 	get: () => createTransposeProgramInfo(gathered, weightTransposeAttribute.perm)
	// });
		context.compute({
		...gatherProgramMetadata,
		cacheHint: attributes.cacheKey,
		get: () => createGatherProgramInfo(inputData, context.inputs[1])
	});
};
export const parseGatherAttributes = (attributes: Record<string, unknown>): GatherAttributes =>
	createAttributeWithCacheKey(attributes as Omit<GatherAttributes, keyof AttributeWithCacheKey>);
