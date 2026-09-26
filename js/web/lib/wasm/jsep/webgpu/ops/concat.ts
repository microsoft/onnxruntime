// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import { DataType } from '../../../wasm-common';
import { TensorView } from '../../tensor-view';
import { ShapeUtil } from '../../util';
import { AttributeWithCacheKey, createAttributeWithCacheKey } from '../attribute-with-cache-key';
import { ComputeContext, ProgramInfo, ProgramInputTensorInfoDependency, ProgramUniform } from '../types';

import { createTensorShapeVariables, IndicesHelper, inputVariable, outputVariable, ShaderHelper } from './common';

export interface ConcatAttributes extends AttributeWithCacheKey {
  readonly axis: number;
}

const validateInputs = (inputs: readonly TensorView[], axis: number): void => {
  if (!inputs || inputs.length < 1) {
    throw new Error('too few inputs');
  }
  const referenceIndex = 0;
  const referenceInput = inputs[referenceIndex];
  const inputType = referenceInput.dataType;
  const inputRank = referenceInput.dims.length;
  inputs.forEach((input, i) => {
    if (i === referenceIndex) {
      return;
    }
    // make sure types of all inputs match
    if (input.dataType !== inputType) {
      throw new Error('input tensors should be one type');
    }
    // make sure the dimensionality of all inputs are the same
    if (input.dims.length !== inputRank) {
      throw new Error('input tensors should have the same shape');
    }
    input.dims.forEach((dim, i) => {
      if (i !== axis && dim !== referenceInput.dims[i]) {
        throw new Error('non concat dimensions must match');
      }
    });
  });
};

const createConcatProgramInfo = (
  inputs: readonly TensorView[],
  adjustedAxis: number,
  outputShape: number[],
  dataType: DataType,
  outputAxisOffset = 0,
): ProgramInfo => {
  const outputSize = inputs.reduce((sum, input) => sum + ShapeUtil.size(input.dims), 0);

  const inputOffsets = new Array<number>(inputs.length);
  const outputAxisOffsets = new Array<number>(inputs.length);
  const inputVars = new Array<IndicesHelper>(inputs.length);

  let inputOffset = 0;
  let axisOffset = outputAxisOffset;
  const inputDependencies: ProgramInputTensorInfoDependency[] = [];
  const inputRanks = [];
  const programUniforms: ProgramUniform[] = [{ type: DataType.uint32, data: outputSize }];
  for (let i = 0; i < inputs.length; ++i) {
    inputOffsets[i] = inputOffset;
    outputAxisOffsets[i] = axisOffset;
    inputOffset += ShapeUtil.size(inputs[i].dims);
    axisOffset += inputs[i].dims[adjustedAxis];
    inputRanks.push(inputs[i].dims.length);
    inputVars[i] = inputVariable(`input${i}`, dataType, inputRanks[i]);
    inputDependencies.push('rank');
    programUniforms.push({ type: DataType.uint32, data: inputOffsets[i] });
    programUniforms.push({ type: DataType.uint32, data: outputAxisOffsets[i] });
  }
  for (let i = 0; i < inputs.length; ++i) {
    programUniforms.push(...createTensorShapeVariables(inputs[i].dims));
  }
  programUniforms.push(...createTensorShapeVariables(outputShape));

  const output = outputVariable('output', dataType, outputShape.length);
  const assignOutputData = inputVars
    .map((input, i) => {
      const inputOffset = `global_idx - uniforms.inputOffset${i}`;
      const assignment = `
        let inputOffset = ${inputOffset};
        var outputIndices = ${input.offsetToIndices('inputOffset')};
        ${output.indicesGet('outputIndices', adjustedAxis)} += uniforms.outputAxisOffset${i};
        ${output.setByIndices('outputIndices', input.getByOffset('inputOffset'))}`;
      if (inputs.length === 1) {
        return assignment;
      }
      if (i === 0) {
        return `if (inputIndex == 0u) {${assignment}}`;
      }
      if (i === inputs.length - 1) {
        return `else {${assignment}}`;
      }
      return `else if (inputIndex == ${i}u) {${assignment}}`;
    })
    .join('\n');
  const getShaderSource = (shaderHelper: ShaderHelper) => `

  ${(() => {
    shaderHelper.registerUniform('outputSize', 'u32');
    for (let i = 0; i < inputs.length; i++) {
      shaderHelper.registerUniform(`inputOffset${i}`, 'u32');
      shaderHelper.registerUniform(`outputAxisOffset${i}`, 'u32');
    }
    return shaderHelper.declareVariables(...inputVars, output);
  })()}

  fn calculateInputIndex(global_idx: u32) -> u32 {
    ${inputVars
      .slice(1)
      .map(
        (_, i) => `if (global_idx < uniforms.inputOffset${i + 1}) {
      return ${i}u;
    }`,
      )
      .join('\n')}
    return ${inputs.length - 1}u;
  }

  ${shaderHelper.mainStart()}
    ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes('uniforms.outputSize')}

    let inputIndex = calculateInputIndex(global_idx);
    ${assignOutputData}
  }`;

  return {
    name: 'Concat',
    shaderCache: { hint: `${adjustedAxis}`, inputDependencies },
    getRunData: () => ({
      outputs: [{ dims: outputShape, dataType }],
      dispatchGroup: { x: Math.ceil(outputSize / 64 /* workgroup size */) },
      programUniforms,
    }),
    getShaderSource,
  };
};

export const concat = (context: ComputeContext, attributes: ConcatAttributes): void => {
  const inputs = context.inputs;
  const inputShape = inputs[0].dims;
  const adjustedAxis = ShapeUtil.normalizeAxis(attributes.axis, inputShape.length);
  validateInputs(inputs, adjustedAxis);
  const outputShape = inputShape.slice();
  outputShape[adjustedAxis] = inputs.reduce(
    (sum, input) => sum + (input.dims.length > adjustedAxis ? input.dims[adjustedAxis] : 0),
    0,
  );
  // 0 length tensors are valid for concat, remove them
  const nonEmptyInputs = inputs.filter((input) => ShapeUtil.size(input.dims) > 0);
  if (nonEmptyInputs.length === 0) {
    context.output(0, outputShape);
    return;
  }

  const maxInputsPerDispatch = context.deviceLimits.maxStorageBuffersPerShaderStage - 1;
  let outputAxisOffset = 0;
  for (let inputIndex = 0; inputIndex < nonEmptyInputs.length; inputIndex += maxInputsPerDispatch) {
    const batchInputs = nonEmptyInputs.slice(inputIndex, inputIndex + maxInputsPerDispatch);
    context.compute(
      createConcatProgramInfo(batchInputs, adjustedAxis, outputShape, inputs[0].dataType, outputAxisOffset),
      {
        inputs: batchInputs,
        outputs: [0],
      },
    );
    outputAxisOffset += batchInputs.reduce((sum, input) => sum + input.dims[adjustedAxis], 0);
  }
};

export const parseConcatAttributes = (attributes: Record<string, unknown>): ConcatAttributes =>
  createAttributeWithCacheKey({ axis: attributes.axis as number });
