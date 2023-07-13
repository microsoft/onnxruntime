// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.


import {DataType} from '../../../wasm-common';
import {TensorView} from '../../tensor';
import {AttributeWithCacheKey, createAttributeWithCacheKey} from '../attribute-with-cache-key';
import {ComputeContext, GpuDataType, ProgramInfo, ProgramInfoLoader, ProgramMetadata} from '../types';
import {ShapeUtil} from '../../util';
import {ShaderHelper} from './common';
import {createTransposeProgramInfo, TransposeAttributes, transposeProgramMetadata} from './transpose';

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

const createGatherProgramInfo = (
	metadata: ProgramMetadata,
	inputData: TensorView,
	index: TensorView,
): ProgramInfo => {
	const inputShape = inputData.dims.slice();
	const indexShape = index.dims.slice();
	const outputShape = new Array(inputShape.length + indexShape.length - 1);

	const outputSize = ShapeUtil.size(outputShape);
	const indexSize = ShapeUtil.size(indexShape);
	const dataType = 'f32';  // TODO: support other data type

	// Step 1: transpose input data along axis

	const getShaderSource = (shaderHelper: ShaderHelper) => `
        @group(0) @binding(0) var<storage, read> inputData : array<${dataType}>;
        @group(0) @binding(1) var<storage, read> index : array<i32>;
        @group(0) @binding(2) var<storage, read_write> output : array<${dataType}>;

        ${shaderHelper.mainStart()}
        	${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes(outputSize)}
        	for (var i : i32 = 0; i < ${indexSize}; i++) {
        	 let idx = index[i];
        	}
    }`;
	return {
		...metadata,
		getShaderSource,
		outputs: [{dims: outputShape, dataType: inputData.dataType, gpuDataType: GpuDataType.default}],
		dispatchGroup: () => ({x: Math.ceil(outputSize / 64 /* workgroup size */)})
	};
};

const createGatherProgramInfoLoader = (
	inputData: TensorView,
	index: TensorView
):
	ProgramInfoLoader => {
	const metadata: ProgramMetadata = {name: 'Gather', inputTypes: [GpuDataType.default]};
	return {
		...metadata,
		get: () => createGatherProgramInfo(
			metadata, inputData, index
		)
	};
};
export const gather = (context: ComputeContext, attributes: GatherAttributes): void => {
	validateInputs(context.inputs);
	const inputShape = context.inputs[0].dims.slice();
	const axis = ShapeUtil.normalizeAxis(attributes.axis, inputShape.length);
	const perm = inputShape;
	perm[axis] = 0;
	perm[0] = axis;
	const weightTransposeAttribute: TransposeAttributes = createAttributeWithCacheKey({perm});
	const transposedInput = context.compute({
			...transposeProgramMetadata,
			cacheHint: weightTransposeAttribute.cacheKey,
			get: () => createTransposeProgramInfo(context.inputs[0], weightTransposeAttribute.perm)
		}
	)[0];

	const gathered = context.compute(
		createGatherProgramInfoLoader(transposedInput, context.inputs[1]))[0];
	context.compute({
		...transposeProgramMetadata,
		cacheHint: weightTransposeAttribute.cacheKey,
		get: () => createTransposeProgramInfo(gathered, weightTransposeAttribute.perm)
	});
};
export const parseGatherAttributes = (attributes: Record<string, unknown>): GatherAttributes =>
	createAttributeWithCacheKey(attributes as Omit<GatherAttributes, keyof AttributeWithCacheKey>);
