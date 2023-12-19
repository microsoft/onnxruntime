// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {tensorDataTypeEnumToString} from '../../../wasm-common';
import {MAX_CLIP, MIN_CLIP} from '../../util';
import {ProgramUniform} from '../types';

import {getWgslMappedType, UniformDataElementType, UniformsArrayType} from './common';

export interface InternalActivationAttributes {
  readonly activation: string;
  readonly clipMin?: number;
  readonly clipMax?: number;
}

export const getActivationSnippet = (attributes: InternalActivationAttributes, valueType: string): string => {
  switch (attributes.activation) {
    case 'Relu':
      return `value = max(value, ${valueType}(0.0));`;
    case 'Sigmoid':
      return `value = (${valueType}(1.0) / (${valueType}(1.0) + exp(-value)));`;
    case 'Clip':
      return `value = clamp(value, ${valueType}(uniforms.clipMin), ${valueType}(uniforms.clipMax));`;
    // TODO: adding other activations that can be fused.
    default:
      return '';
  }
};

export const parseInternalActivationAttributes =
    (attributes: Record<string, unknown>|undefined): InternalActivationAttributes => {
      const activation = attributes?.activation as string || '';

      if (activation === 'Clip') {
        const [clipMin, clipMax] = attributes?.activation_params as [number, number] || [MIN_CLIP, MAX_CLIP];
        return {activation, clipMax, clipMin};
      }
      return {activation};
    };

export const updateUniformsFromActivation =
    (programUniforms: ProgramUniform[], uniforms: UniformsArrayType, attributes: InternalActivationAttributes,
     dataType: number) => {
      const tensorDataType = tensorDataTypeEnumToString(dataType) as ProgramUniform['type'];
      const wgslElementType = getWgslMappedType(dataType, 1);
      if (typeof wgslElementType !== 'string') {
        throw new Error(`clipMax and clipMin doesn't support type ${wgslElementType[0]}!`);
      }
      if (attributes.activation === 'Clip') {
        programUniforms.push(
            {type: tensorDataType, data: attributes.clipMax!}, {type: tensorDataType, data: attributes.clipMin!});
        uniforms.push(
            {name: 'clipMax', type: wgslElementType as UniformDataElementType},
            {name: 'clipMin', type: wgslElementType as UniformDataElementType});
      }
    };
