// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {DataType} from '../../../wasm-common';
import {TensorView} from '../../tensor-view';
import {ShapeUtil} from '../../util';
import {AttributeWithCacheKey, createAttributeWithCacheKey} from '../attribute-with-cache-key';
import {ComputeContext, ProgramInfo, ProgramUniform} from '../types';

import {createTensorShapeVariables, IndicesHelper, inputVariable, outputVariable, ShaderHelper, UniformsArrayType} from './common';

export interface GatherBlockQuantizedAttributes extends AttributeWithCacheKey {
  blockSize: number;
  gatherAxis: number;
  quantizeAxis: number;
}
export const validateInputs = (inputs: readonly TensorView[], attributes: GatherBlockQuantizedAttributes): void => {
  if (inputs.length < 3 || inputs.length > 4) {
    throw new Error('GatherBlockQuantized requires 3 or 4 inputs.');
  }
  const quantizeAxis = ShapeUtil.normalizeAxis(attributes.quantizeAxis, inputs[0].dims.length);
  const blockSize = attributes.blockSize;
  const data = inputs[0];
  const scales = inputs[2];
  const zeroPoint = inputs.length === 4 ? inputs[3] : undefined;
  if (scales.dims.length !== data.dims.length ||
      data.dims.map((d, i) => i === quantizeAxis ? Math.ceil(d / blockSize) === scales.dims[i] : d === scales.dims[i])
          .reduce((a, b) => a && b, true)) {
    throw new Error('Scales must have the same rank as the input tensor.');
  }
  if (zeroPoint) {
    if (zeroPoint.dataType !== data.dataType) {
      throw new Error('Zero point must have the same data type as the input tensor.');
    }
    if (zeroPoint.dims.length !== scales.dims.length ||
       zeroPoint.dims.map((d, i) => d === scales.dims[i]).reduce((a, b) => a && b, true)) {
      throw new Error('Zero point must have the same rank as the input tensor.');
    }
  }
};

const createGatherBlockQuantizedProgramInfo =
    (inputs: readonly TensorView[], attributes: GatherBlockQuantizedAttributes): ProgramInfo => {
      const inputShape = inputs[0].dims;
      const indicesShape = inputs[1].dims;
      const inputRank = inputShape.length;
      const gatherAxis = ShapeUtil.normalizeAxis(attributes.gatherAxis, inputRank);
      const quantizeAxis = ShapeUtil.normalizeAxis(attributes.quantizeAxis, inputRank);
      const outputShape = inputShape.slice(0);
      outputShape.splice(gatherAxis, 1, ...indicesShape);
      const outputSize = ShapeUtil.size(outputShape);
      const outputType = inputs[2].dataType;
      const programUniforms: ProgramUniform[] = [
        {type: DataType.uint32, data: outputSize}, {type: DataType.int32, data: quantizeAxis},
        {type: DataType.uint32, data: gatherAxis},
        ...createTensorShapeVariables(inputs[0].dims, inputs[1].dims, outputShape)
      ];

      const getShaderSource = (shaderHelper: ShaderHelper) => {
        const data = inputVariable('data', inputs[0].dataType, inputs[0].dims.length);
        const indices = inputVariable('inputIndices', inputs[1].dataType, inputs[1].dims.length);
        const scales = inputVariable('scales', inputs[2].dataType, inputs[2].dims.length);
        const zeroPoint =
            (inputs.length > 3) ? inputVariable('zeroPoint', inputs[3].dataType, inputs[3].dims.length) : undefined;
        const output = outputVariable('output', outputType, outputShape.length);
        const inputVariables = [data, indices, scales];
        if (zeroPoint) {
          inputVariables.push(zeroPoint);
        }
        const uniforms: UniformsArrayType = [
          {name: 'output_size', type: 'u32'}, {name: 'quantize_axis', type: 'i32'}, {name: 'gather_axis', type: 'u32'}
        ];
        return `
        ${shaderHelper.registerUniforms(uniforms).declareVariables(...inputVariables, output)}
        ${shaderHelper.mainStart()}
    }`;
      };
      return {
        name: 'GatherBlockQuantized',
        shaderCache:
            {hint: attributes.cacheKey, inputDependencies: Array.from({length: inputs.length}, (_v, _i) => 'rank')},
        getRunData: () => ({
          outputs: [
            {dims: outputShape, dataType: outputType},
          ],
          dispatchGroup: {x: Math.ceil(outputSize / 64 /* workgroup size */)},
          programUniforms
        }),
        getShaderSource,
      };
    };

export const gatherBlockQuantized = (context: ComputeContext, attributes: GatherBlockQuantizedAttributes): void => {
  const inputs = context.inputs;
  validateInputs(inputs, attributes);
  context.compute(createGatherBlockQuantizedProgramInfo(context.inputs, attributes));
};

export const parseGatherBlockQuantizedAttributes =
    (attributes: Record<string, unknown>): GatherBlockQuantizedAttributes => createAttributeWithCacheKey({
      blockSize: attributes.blockSize as number,
      gatherAxis: attributes.gatherAxis as number,
      quantizeAxis: attributes.quantizeAxis as number
    });
