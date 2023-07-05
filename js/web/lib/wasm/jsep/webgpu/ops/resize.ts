// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.


import {DataType} from '../../../wasm-common';
import {TensorView} from '../../tensor';
import {ShapeUtil} from '../../util';
import {AttributeWithCacheKey, createAttributeWithCacheKey} from '../attribute-with-cache-key';
import {ComputeContext, GpuDataType, ProgramInfo, ProgramInfoLoader, ProgramMetadata} from '../types';

import {createIndicesHelper, ShaderHelper} from './common';

export interface ResizeAttributes extends AttributeWithCacheKey {
  antialias: number;
  axes: number[];
  coordinateTransformationMode: string;
  cubicCoeffA: number;
  excludeOutsize: number;
  extrapolationValue: number;
  keepAspectRatioPolicy: string;
  mode: string;
  nearestMode: string;
}

const validateInputs = (inputs: readonly TensorView[], attributes: ResizeAttributes): void => {
  const rank = inputs[0].dims.length;
  if (inputs.length > 1) {
    if (inputs[1].dims.length !== 2 * rank) {
      throw new Error('Resize requires RoI input to be of rank 2*rank');
    }
  }
  if (inputs.length > 2) {
    if (inputs[2].dataType === DataType.float) {
      // The input is scales
      if (attributes.axes.length > 0) {
        if (inputs[2].dims.length !== attributes.axes.length) {
          throw new Error('Resize requires "scales" input size to be of axes rank when axes attributes is specified');
        }
      } else {
        if (inputs[2].dims.length !== rank) {
          throw new Error('Resize requires scales size to be of input rank');
        }
      }
    } else if (inputs[2].dataType === DataType.int64) {
      // The input is sizes
      if (attributes.axes.length > 0) {
        if (inputs[2].dims.length !== attributes.axes.length) {
          throw new Error(
              'Resize requires "sizes" input size to be of rank axes rank when axes attributes is specified');
        }
      } else if (inputs[2].dims.length !== rank) {
        throw new Error('Resize requires sizes size to be of rank input rank');
      }
    }
  }
};
const createResizeProgramInfo =
    (metadata: ProgramMetadata, inputs: readonly TensorView[], attributes: ResizeAttributes): ProgramInfo => {
      const inputShape = inputs[0].dims;
      const outputShape: number[] = [];
      const roi: number[] = [];
      const scales: number[] = [];
      const sizes: number[] = [];

      if (inputs.length > 1) {
        inputs[1].getFloat32Array().forEach(value => roi.push(value));
      }
      if (inputs.length > 2) {
        if (inputs[2].dataType === DataType.float) {
          inputs[2].getFloat32Array().forEach(value => scales.push(value));
        } else if (inputs[2].dataType === DataType.int64) {
          inputs[2].getBigInt64Array().forEach(value => sizes.push(Number(value)));
        }
      }

      if (sizes.length > 0) {
        sizes.forEach(value => scales.push(value));
      } else if (roi.length > 0 && scales.length > 0) {
        for (let i = 0; i < inputs[0].dims.length; i++) {
          outputShape.push(Math.floor(inputs[0].dims[i] * (roi[2 * i + 1] - roi[2 * i]) * scales[i]));
        }
      } else {
        inputShape.forEach(value => outputShape.push(value));
      }

      const outputIndicesHelper = createIndicesHelper('output', outputShape);
      const inputIndicesHelper = createIndicesHelper('input', inputShape);
      const outputSize = ShapeUtil.size(outputShape);
      const dataType = 'f32';
      const getShaderSource = (shaderHelper: ShaderHelper) => `
      @group(0) @binding(0) var<storage, read> input : array<${dataType}>;
      @group(0) @binding(1) var<storage, read_write> output : array<${dataType}>;
      ${outputIndicesHelper.o2iImpl}
      ${inputIndicesHelper.i2oImpl}
      ${shaderHelper.mainStart()}
        ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes(outputSize)}

        ${outputIndicesHelper.indicesVariableDeclaration('indices')}
        ${outputIndicesHelper.o2iCall('global_idx', 'indices')}
        ${inputIndicesHelper.indicesVariableDeclaration('aIndices')}

        output[global_idx] = input[${inputIndicesHelper.i2oExpression('inputIndices')}];
      }`;

      return {
        ...metadata,
        getShaderSource,
        outputs: [{dims: outputShape, dataType: inputs[0].dataType, gpuDataType: GpuDataType.default}],
        dispatchGroup: () => ({x: Math.ceil(outputSize / 64 /* workgroup size */)})
      };
    };

export const createResizeProgramInfoLoader =
    (inputs: readonly TensorView[], attributes: ResizeAttributes): ProgramInfoLoader => {
      const metadata: ProgramMetadata = {
        name: 'Resize',
        inputTypes: [GpuDataType.default],
        cacheHint: attributes.cacheKey,
      };
      return {...metadata, get: () => createResizeProgramInfo(metadata, inputs, attributes)};
    };

export const resize = (context: ComputeContext, attributes: ResizeAttributes): void => {
  validateInputs(context.inputs, attributes);
  context.compute(createResizeProgramInfoLoader(context.inputs, attributes), {inputs: [0]});
};

export const parseResizeAttributes = (attributes: Record<string, unknown>): ResizeAttributes =>
    createAttributeWithCacheKey(attributes as Omit<ResizeAttributes, keyof AttributeWithCacheKey>);
