// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {DataType} from '../../../wasm-common';
import {TensorView} from '../../tensor-view';
import {ShapeUtil} from '../../util';
import {AttributeWithCacheKey, createAttributeWithCacheKey} from '../attribute-with-cache-key';
import {ComputeContext, ProgramInfo} from '../types';

import {castToF32, fillVector, getMaxComponents, inputVariable, outputVariable, ShaderHelper, sumVector, tensorTypeToWsglStorageType,} from './common';

export interface SkipLayerNormAttributes extends AttributeWithCacheKey {
  epsilon: number;
}

const validateInputs = (inputs: readonly TensorView[]): void => {
  if (!inputs || inputs.length < 3) {
    throw new Error('layerNorm requires at least 3 inputs.');
  }

  const input: TensorView = inputs[0];
  const skip: TensorView = inputs[1];
  const gamma: TensorView = inputs[2];

  if (input.dataType !== skip.dataType || input.dataType !== gamma.dataType) {
    throw new Error('All inputs must have the same data type');
  }

  if (input.dims.length !== 3 && input.dims.length !== 2) {
    throw new Error('Input must be 2D or 3D');
  }

  if (skip.dims.length !== 3 && skip.dims.length !== 2) {
    throw new Error('Skip must be 2D or 3D');
  }

  const hiddenSize = input.dims[input.dims.length - 1];
  const sequenceLength = input.dims[input.dims.length - 2];
  if (skip.dims[skip.dims.length - 1] !== hiddenSize) {
    throw new Error('Skip must have the same hidden size as input');
  }
  if (skip.dims[skip.dims.length - 2] !== sequenceLength) {
    throw new Error('Skip must have the same sequence length as input');
  }

  if (gamma.dims.length !== 1) {
    throw new Error('Gamma must be 1D');
  }
  if (gamma.dims[gamma.dims.length - 1] !== hiddenSize) {
    throw new Error('Gamma must have the same hidden size as input');
  }
  if (inputs.length > 3) {
    const beta: TensorView = inputs[3];
    if (beta.dims.length !== 1) {
      throw new Error('Beta must be 1D');
    }
    if (beta.dims[beta.dims.length - 1] !== hiddenSize) {
      throw new Error('Beta must have the same hidden size as input');
    }
  }

  if (inputs.length > 4) {
    const bias: TensorView = inputs[4];
    if (bias.dims.length !== 1) {
      throw new Error('Bias must be 1D');
    }
    if (bias.dims[bias.dims.length - 1] !== hiddenSize) {
      throw new Error('Bias must have the same hidden size as input');
    }
  }
};

const createSkipLayerNormProgramInfo =
    (inputs: readonly TensorView[], attributes: SkipLayerNormAttributes, outputCount: number, isTraining: boolean):
        ProgramInfo => {
          const inputShape = inputs[0].dims;
          const inputSize = ShapeUtil.size(inputShape);
          const outputShape = inputShape;
          const outputSize = inputSize;
          const hiddenSize = inputShape.slice(-1)[0];
          const meanInvStdDevDim = isTraining ? inputShape.slice(0, -1).concat(1) : [];
          const hasBetaInput = inputs.length > 3;
          const hasBiasInput = inputs.length > 4;
          const hasMeanOutput = isTraining && outputCount > 1;
          const hasInvStdDevOutput = isTraining && outputCount > 2;
          const hasInputSkipBiasSumOutput = outputCount > 3;

          const components = getMaxComponents(hiddenSize);
          const variables = [
            inputVariable('x', inputs[0].dataType, inputs[0].dims, components),
            inputVariable('skip', inputs[1].dataType, inputs[1].dims, components),
            inputVariable('gamma', inputs[2].dataType, inputs[2].dims, components),
          ];
          if (hasBetaInput) {
            variables.push(inputVariable('beta', inputs[3].dataType, inputs[3].dims, components));
          }
          if (hasBiasInput) {
            variables.push(inputVariable('bias', inputs[4].dataType, inputs[4].dims, components));
          }
          variables.push(outputVariable('output', inputs[0].dataType, outputShape, components));
          if (hasMeanOutput) {
            variables.push(outputVariable('meanOutput', DataType.float, meanInvStdDevDim));
          }
          if (hasInvStdDevOutput) {
            variables.push(outputVariable('invStdOutput', DataType.float, meanInvStdDevDim));
          }
          if (hasInputSkipBiasSumOutput) {
            variables.push(outputVariable('inputSkipBiasSum', inputs[0].dataType, outputShape, components));
          }
          const dataType = tensorTypeToWsglStorageType(inputs[0].dataType);
          const getShaderSource = (shaderHelper: ShaderHelper) => `
      const hiddenSize: f32 = ${hiddenSize};
      const hiddenSizeVectorized: u32 = ${hiddenSize / components};
      const epsilon: f32 = ${attributes.epsilon};

      ${shaderHelper.declareVariables(...variables)}

      ${shaderHelper.mainStart()}
        ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes(outputSize / hiddenSize)}
        let offset = global_idx * hiddenSizeVectorized;
        var sum = ${fillVector('f32', components)};
        var squareSum = ${fillVector('f32', components)};
        for (var i: u32 = 0; i < hiddenSizeVectorized; i++) {
          let skipValue = skip[offset + i];
          let biasValue = ${hasBiasInput ? 'bias[i]' : '0.0'};
          let inputValue = x[offset + i];
          let value = inputValue + skipValue + biasValue;
          ${hasInputSkipBiasSumOutput ? 'inputSkipBiasSum[offset + i] = value;' : ''}
          output[offset + i] = value;
          let f32Value = ${castToF32(dataType, components, 'value')};
          sum += f32Value;
          squareSum += f32Value * f32Value;
        }
        let mean = ${sumVector('sum', components)} / hiddenSize;
        let variance = sqrt(${sumVector('squareSum', components)} / hiddenSize - mean * mean + epsilon);
        ${hasMeanOutput ? 'meanOutput[global_idx] = mean;' : ''}
        ${hasInvStdDevOutput ? 'invStdOutput[global_idx] = 1.0 / variance;' : ''}
        for (var i: u32 = 0; i < hiddenSizeVectorized; i++) {
          output[offset + i] = (output[offset + i] - ${dataType}(mean)) / ${dataType}(variance) * gamma[i]
           + ${hasBetaInput ? 'beta[i]' : '0.0'};
        }
      }`;
          const outputs = [{dims: outputShape, dataType: inputs[0].dataType}];
          if (outputCount > 1) {
            outputs.push({dims: meanInvStdDevDim, dataType: DataType.float});
          }
          if (outputCount > 2) {
            outputs.push({dims: meanInvStdDevDim, dataType: DataType.float});
          }
          if (outputCount > 3) {
            outputs.push({dims: inputShape, dataType: inputs[0].dataType});
          }

          return {
            name: 'SkipLayerNormalization',
            shaderCache: {hint: attributes.cacheKey},
            getShaderSource,
            getRunData: () => ({outputs, dispatchGroup: {x: Math.ceil(outputSize / hiddenSize / 64)}}),
          };
        };

export const skipLayerNorm = (context: ComputeContext, attributes: SkipLayerNormAttributes): void => {
  // TODO: initialize isTraining from ComputeContext
  const isTraining = false;
  validateInputs(context.inputs);
  // Mean and InvStdDev are only used in training mode and are not required for inference.
  // They are added here for completeness only.
  const outputs = [0];
  if (context.outputCount > 1) {
    outputs.push(isTraining ? 1 : -3);
  }
  if (context.outputCount > 2) {
    outputs.push(isTraining ? 2 : -3);
  }
  if (context.outputCount > 3) {
    outputs.push(3);
  }
  context.compute(
      createSkipLayerNormProgramInfo(context.inputs, attributes, context.outputCount, isTraining), {outputs});
};

export const parseSkipLayerNormAttributes = (attributes: Record<string, unknown>): SkipLayerNormAttributes => {
  const epsilon = attributes.epsilon as number;
  return createAttributeWithCacheKey({epsilon});
};
