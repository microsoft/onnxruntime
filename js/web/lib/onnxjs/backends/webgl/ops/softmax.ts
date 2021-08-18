// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {AttributeWithCacheKey, createAttributeWithCacheKey} from '../../../attribute-with-cache-key';
import {Graph} from '../../../graph';
import {OperatorImplementation, OperatorInitialization} from '../../../operators';
import {Tensor} from '../../../tensor';
import {ShapeUtil} from '../../../util';
import {getGlsl} from '../glsl-source';
import {WebGLInferenceHandler} from '../inference-handler';
import {ProgramInfo, TextureType} from '../types';

export interface SoftmaxAttributes extends AttributeWithCacheKey {
  readonly axis: number;
}

const softmaxComputeMaxProgramMetadata = {
  name: 'SoftmaxComputeMax',
  inputNames: ['A'],
  inputTypes: [TextureType.unpacked],
};

const softmaxComputeScaleProgramMetadata = {
  name: 'SoftmaxComputeScale',
  inputNames: ['A', 'Max'],
  inputTypes: [TextureType.unpacked, TextureType.unpacked],
};

const softmaxProgramMetadata = {
  name: 'SoftMax',
  inputNames: ['A', 'Max', 'Norm'],
  inputTypes: [TextureType.unpacked, TextureType.unpacked, TextureType.unpacked],
};

export const softmax: OperatorImplementation<SoftmaxAttributes> =
    (inferenceHandler: WebGLInferenceHandler, inputs: Tensor[], attributes: SoftmaxAttributes): Tensor[] => {
      validateInputs(inputs);

      const inputShape = inputs[0].dims.slice();
      const axis = ShapeUtil.normalizeAxis(attributes.axis, inputShape.length);
      const N = ShapeUtil.sizeToDimension(inputShape, axis);
      const D = ShapeUtil.sizeFromDimension(inputShape, axis);

      const computeMaxProgramInfo = createComputeMaxProgramInfo(inferenceHandler, inputs[0], N, D, [N]);
      const max = inferenceHandler.run(
          {...softmaxComputeMaxProgramMetadata, cacheHint: attributes.cacheKey, get: () => computeMaxProgramInfo},
          inputs);

      const computeScaleProgramInfo =
          createComputScaleProgramInfo(inferenceHandler, inputs[0], N, D, computeMaxProgramInfo.output.dims, [N]);
      const scale = inferenceHandler.run(
          {...softmaxComputeScaleProgramMetadata, cacheHint: attributes.cacheKey, get: () => computeScaleProgramInfo},
          [inputs[0], max]);

      const softMaxProgramInfo = createSoftMaxProgramInfo(
          inferenceHandler, inputs[0], N, D, computeMaxProgramInfo.output.dims, computeScaleProgramInfo.output.dims);
      const output = inferenceHandler.run(
          {...softmaxProgramMetadata, cacheHint: attributes.cacheKey, get: () => softMaxProgramInfo},
          [inputs[0], max, scale]);
      return [output];
    };

export const parseSoftmaxAttributes: OperatorInitialization<SoftmaxAttributes> =
    (node: Graph.Node): SoftmaxAttributes => createAttributeWithCacheKey({axis: node.attributes.getInt('axis', 1)});

/**
 * Create a texture that contains the maximum value of each of the 'N' rows
 */
const createComputeMaxProgramInfo =
    // eslint-disable-next-line @typescript-eslint/naming-convention
    (inferenceHandler: WebGLInferenceHandler, input: Tensor, N: number, D: number, outputShape: number[]):
        ProgramInfo => {
          const [textureWidth, textureHeight] =
              inferenceHandler.calculateTextureWidthAndHeight(input.dims, TextureType.unpacked);
          const rank = outputShape.length;

          if (N < 1 || D < 1) {
            throw new Error('Logical row count N and feature count D must be greater than or equal to 1');
          }

          if (outputShape.length !== 1) {
            throw new Error('Dimensionality of the output should be 1');
          }

          if (outputShape[0] !== N) {
            throw new Error('Shape of the output should be equal to logical row count');
          }

          const glsl = getGlsl(inferenceHandler.session.backend.glContext.version);
          const shaderSource = `
      float process(int[${rank}] indices) {
        int logical_row_start_offset = indices[0] * ${D};

        float max = getColorAsFloat(${glsl.texture2D}(A, offsetToCoords(logical_row_start_offset, ${textureWidth},
        ${textureHeight} )));
        for(int i=1; i<${D}; ++i)
        {
          float current = getColorAsFloat(${glsl.texture2D}(A, offsetToCoords(logical_row_start_offset + i,
            ${textureWidth}, ${textureHeight})));
          if(current > max)
          max = current;
        }

        return max;
      }`;
          return {
            ...softmaxComputeMaxProgramMetadata,
            output: {dims: outputShape, type: input.type, textureType: TextureType.unpacked},
            shaderSource
          };
        };

/**
 * Create a texture that contains the normalization factor for each of the 'N' rows
 */
const createComputScaleProgramInfo =
    // eslint-disable-next-line @typescript-eslint/naming-convention
    (inferenceHandler: WebGLInferenceHandler, input: Tensor, N: number, D: number,
     maxElementPerLogicalRow: readonly number[], outputShape: number[]): ProgramInfo => {
      const [textureWidth, textureHeight] =
          inferenceHandler.calculateTextureWidthAndHeight(input.dims, TextureType.unpacked);
      const rank = outputShape.length;

      if (N < 1 || D < 1) {
        throw new Error('Logical row count N and feature count D must be greater than or equal to 1');
      }

      if (outputShape.length !== 1) {
        throw new Error('Dimensionality of the output should be 1');
      }

      if (outputShape[0] !== N) {
        throw new Error('Shape of the output should be equal to logical row count');
      }

      if (maxElementPerLogicalRow.length !== 1) {
        throw new Error('Dimensionality of the intermediate results should be 1');
      }

      if (maxElementPerLogicalRow[0] !== N) {
        throw new Error('Shape of the intermediate results should be equal to logical row count');
      }

      const glsl = getGlsl(inferenceHandler.session.backend.glContext.version);
      const shaderSource = `
      float process(int[${rank}] indices) {
        int logical_row_start_offset = indices[0] * ${D};

        float norm_factor = 0.0;
        float max = _Max(indices);
        for(int i=0; i<${D}; ++i)
        {
          norm_factor += exp(getColorAsFloat(${glsl.texture2D}(A, offsetToCoords(logical_row_start_offset + i,
            ${textureWidth}, ${textureHeight}))) - max);
        }

        return norm_factor;
      }`;
      return {
        ...softmaxComputeScaleProgramMetadata,
        output: {dims: outputShape, type: input.type, textureType: TextureType.unpacked},
        shaderSource
      };
    };

const createSoftMaxProgramInfo =
    // eslint-disable-next-line @typescript-eslint/naming-convention
    (inferenceHandler: WebGLInferenceHandler, input: Tensor, N: number, D: number,
     maxElementPerLogicalRow: readonly number[], normalizationPerLogicalRow: readonly number[]): ProgramInfo => {
      const [textureWidth, textureHeight] =
          inferenceHandler.calculateTextureWidthAndHeight(input.dims, TextureType.unpacked);
      const rank = input.dims.length;

      if (N < 1 || D < 1) {
        throw new Error('Logical row count N and feature count D must be greater than or equal to 1');
      }

      if (maxElementPerLogicalRow.length !== 1 || normalizationPerLogicalRow.length !== 1) {
        throw new Error('Dimensionality of the intermediate results should be 1');
      }

      if (maxElementPerLogicalRow[0] !== N || normalizationPerLogicalRow[0] !== N) {
        throw new Error('Shape of the intermediate results should be equal to logical row count');
      }

      const shaderSource = `
      float process(int[${rank}] indices) {

      // get offset of current logical tensor index from the 2-D texture coordinates (TexCoords)
      int offset = coordsToOffset(TexCoords, ${textureWidth}, ${textureHeight});

      //determine the logical row for this index
      int logical_row_index[1];
      logical_row_index[0] = offset / ${D};

      float norm_factor = _Norm(logical_row_index);

      // avoid possible division by 0
      // if norm_facor is 0, all elements are zero
      // if so, return 0
      if(norm_factor == 0.0)
        return 0.0;

      return exp(_A(indices) - _Max(logical_row_index)) / norm_factor;
    }`;
      return {
        ...softmaxProgramMetadata,
        output: {dims: input.dims, type: input.type, textureType: TextureType.unpacked},
        shaderSource
      };
    };

const validateInputs = (inputs: Tensor[]): void => {
  if (!inputs || inputs.length !== 1) {
    throw new Error('Softmax requires 1 input.');
  }

  if (inputs[0].type !== 'float32' && inputs[0].type !== 'float64') {
    throw new Error('Invalid input type');
  }
};