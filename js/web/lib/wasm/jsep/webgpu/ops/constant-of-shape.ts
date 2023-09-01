// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {ShapeUtil} from '../../util';
import {AttributeWithCacheKey, createAttributeWithCacheKey} from '../attribute-with-cache-key';
import {ComputeContext, GpuDataType, ProgramInfo, ProgramMetadata} from '../types';

import {outputVariable, ShaderHelper} from './common';

export interface ConstantOfShapeAttributes extends AttributeWithCacheKey {
  readonly value: number;
  readonly dataType: number;
}

const createConstantOfShapeProgramInfo =
    (metadata: ProgramMetadata, outputDims: number[], attributes: ConstantOfShapeAttributes): ProgramInfo => {
      const outputSize = ShapeUtil.size(outputDims);
      const output = outputVariable('output', attributes.dataType, outputDims);

      const getShaderSource = (shaderHelper: ShaderHelper) => `
        ${shaderHelper.declareVariables(output)}
        ${shaderHelper.mainStart()}
        ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes(outputSize)}
        ${output.setByOffset('global_idx', attributes.value.toString())}
      }`;

      return {
        ...metadata,
        outputs: [{dims: outputDims, dataType: attributes.dataType, gpuDataType: GpuDataType.default}],
        getShaderSource,
        dispatchGroup: () => ({x: Math.ceil(outputSize / 64 /* workgroup size */)})
      };
    };

export const constantOfShape = (context: ComputeContext, attributes: ConstantOfShapeAttributes): void => {
  const outputDims = Array.from(context.inputs[0].getBigInt64Array(), Number);
  const metadata: ProgramMetadata = {name: 'ConstantOfShape', inputTypes: [], cacheHint: attributes.cacheKey};
  context.compute(
      {...metadata, get: () => createConstantOfShapeProgramInfo(metadata, outputDims, attributes)}, {inputs: []});
};

export const parseConstantOfShapeAttributes = (attributes: Record<string, unknown>): ConstantOfShapeAttributes =>
    createAttributeWithCacheKey({value: attributes.value as number, dataType: attributes.dataType as number});
