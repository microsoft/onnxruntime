// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {TensorView} from '../../tensor-view';
import {ShapeUtil} from '../../util';
import {AttributeWithCacheKey, createAttributeWithCacheKey} from '../attribute-with-cache-key';
import {ComputeContext, ProgramInfo} from '../types';

import {inputVariable, outputVariable, ShaderHelper} from './common';

export interface BatchNormAttributes extends AttributeWithCacheKey {
  readonly epsilon: number;
  readonly momentum: number;
  readonly spatial: boolean;
  readonly trainingMode: boolean;
  readonly format: 'nhwc'|'nchw';
  readonly outputCount: number;
}

const validateInputs = (inputs: readonly TensorView[], attributes: BatchNormAttributes): void => {
  if (!inputs || inputs.length !== 5) {
    throw new Error('BatchNormalization requires 5 inputs');
  }

  const checkShapeEqual = (actual: readonly number[], expected: readonly number[], message: string) => {
    const r = expected.length;
    if (r !== actual.length) {
      throw new Error(`${message}: num dimensions != ${r}`);
    }
    expected.forEach((v, i) => {
      if (v !== actual[i]) {
        switch (i % 100) {
          case 11:
          case 12:
          case 13:
            throw new Error(`${message}: ${i}th dimension != ${v}`);
          default:
        }
        switch (i % 10) {
          case 1:
            throw new Error(`${message}: ${i}st dimension != ${v}`);
          case 2:
            throw new Error(`${message}: ${i}nd dimension != ${v}`);
          case 3:
            throw new Error(`${message}: ${i}rd dimension != ${v}`);
          default:
            throw new Error(`${message}: ${i}th dimension != ${v}`);
        }
      }
    });
  };

  if (inputs[0].dims.length > 1) {
    const shape = Object.seal<Record<typeof attributes.format, number[]>>({
      nhwc: inputs[0].dims.slice(-1),
      nchw: inputs[0].dims.slice(1, attributes.spatial ? 2 : undefined),
    })[attributes.format];
    checkShapeEqual(inputs[1].dims, shape, 'Invalid input scale');
    checkShapeEqual(inputs[2].dims, shape, 'Invalid input B');
    checkShapeEqual(inputs[3].dims, shape, 'Invalid input mean');
    checkShapeEqual(inputs[4].dims, shape, 'Invalid input var');
  } else {
    checkShapeEqual(inputs[1].dims, [1], 'Invalid input scale');
    checkShapeEqual(inputs[2].dims, [1], 'Invalid input B');
    checkShapeEqual(inputs[3].dims, [1], 'Invalid input mean');
    checkShapeEqual(inputs[4].dims, [1], 'Invalid input var');
  }
};

const createBatchNormInferenceProgramInfo = (inputs: readonly TensorView[], attributes: BatchNormAttributes):
    ProgramInfo => {
      const {epsilon, spatial, format} = attributes;
      const yShape = inputs[0].dims;
      const outputSize = ShapeUtil.size(yShape);
      const x = inputVariable('x', inputs[0].dataType, inputs[0].dims);
      const scale = inputVariable('scale', inputs[1].dataType, [ShapeUtil.size(inputs[1].dims)]);
      const bias = inputVariable('bias', inputs[2].dataType, [ShapeUtil.size(inputs[2].dims)]);
      const inputMean = inputVariable('inputMean', inputs[3].dataType, [ShapeUtil.size(inputs[3].dims)])
      const inputVar = inputVariable('inputVar', inputs[4].dataType, [ShapeUtil.size(inputs[4].dims)]);
      const y = outputVariable('y', inputs[0].dataType, yShape);

      const calcCOffset = (): string => {
        let cOffset = '';
        if (spatial) {
          cOffset = `let cOffset = ${
              yShape.length === 1   ? '0u' :
                  format === 'nhwc' ? `outputIndices[${yShape.length - 1}]` :
                                      `outputIndices[1]`};`;
        } else {
          if (format === 'nchw') {
            cOffset = `
            ${y.indicesSet('outputIndices', '0', '0')}
            let cOffset = ${y.indicesToOffset('outputIndices')};`;
          } else {
            // update C channel.
            cOffset = `var cIndices = ${scale.type.indices}('0');
                       cIndices[0] = outputIndices[${yShape.length - 1}];`;
            // update D1 x ... x Dn channels.
            for (let i = 1; i < scale.rank; i++) {
              cOffset += `cIndices[${i}] = outputIndices[${i + 1}];`;
            }
            cOffset += `let cOffset = ${scale.indicesToOffset('cIndices')};`;
          }
        }
        return cOffset;
      };
      const getInferenceModeShaderSource = (helper: ShaderHelper) => `
  const epsilon = ${epsilon};
  ${helper.declareVariables(x, scale, bias, inputMean, inputVar, y)}
  ${helper.mainStart()}
  ${helper.guardAgainstOutOfBoundsWorkgroupSizes(outputSize)}
    var outputIndices = ${y.offsetToIndices('global_idx')};
    ${calcCOffset()}
    let scale = ${scale.getByOffset('cOffset')};
    let bias = ${bias.getByOffset('cOffset')};
    let inputMean = ${inputMean.getByOffset('cOffset')};
    let inputVar = ${inputVar.getByOffset('cOffset')};
    let x = ${x.getByOffset('global_idx')};
    let value = (x - inputMean) / sqrt(inputVar + epsilon) * scale + bias;
    ${y.setByOffset('global_idx', 'value')}
  }`;
      return {
        name: 'BatchNormalization',
        shaderCache: {hint: `${attributes.epsilon}_${attributes.format}_${spatial}`},
        getShaderSource: getInferenceModeShaderSource,
        getRunData: () => ({
          outputs: [{dims: inputs[0].dims, dataType: inputs[0].dataType}],
          dispatchGroup: {x: Math.ceil(outputSize / 64 /* workgroup size */)},
        }),
      };
    }

export const parseBatchNormAttributes = (attributes: Record<string, unknown>): BatchNormAttributes =>
    createAttributeWithCacheKey(attributes as Omit<BatchNormAttributes, keyof AttributeWithCacheKey>);

export const batchNorm = (context: ComputeContext, attributes: Record<string, unknown>): void => {
  const {inputs, outputCount} = context;
  const updatedAttributes = parseBatchNormAttributes({...attributes, outputCount});
  validateInputs(inputs, updatedAttributes);
  if (attributes.trainingMode) {
    throw new Error('BatchNormalization trainingMode is not supported yet.');
  } else {
    context.compute(createBatchNormInferenceProgramInfo(inputs, updatedAttributes));
  }
};
