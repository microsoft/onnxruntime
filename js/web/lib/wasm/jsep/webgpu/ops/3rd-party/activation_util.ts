/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

// sampled from [@tensorflow/tfjs] tfjs-backend-webgpu/src/activation_util.ts
//
// modified to fit the needs of the project
import {BinaryOpType, getBinaryOpString} from './binary_op_util';
import {getUnaryOpString, UnaryOpType} from './unary_op_util';

export declare type Activation = 'linear' | 'relu' | 'prelu' | 'elu' | 'relu6' | 'leakyrelu' | 'sigmoid' | 'gelu';

export const typeSnippet = (component: number, dataType: string) => {
  switch (component) {
    case 1:
      return dataType;
    case 2:
      return `vec2<${dataType}>`;
    case 3:
      return `vec3<${dataType}>`;
    case 4:
      return `vec4<${dataType}>`;
    default:
      throw new Error(`${component}-component is not supported.`);
  }
};
export const activationFnSnippet =
    (activation?: Activation, _hasPreluActivationWeights = false, _packed = false, _coordsLength = 3): string => {
      if (!activation) {
        return '';
      }

      let activationOpSnippet = '';
      if (activation === 'linear') {
        activationOpSnippet = getUnaryOpString(UnaryOpType.LINEAR);
      } else if (activation === 'relu') {
        activationOpSnippet = getUnaryOpString(UnaryOpType.RELU, _packed);
      } else if (activation === 'elu') {
        activationOpSnippet = getUnaryOpString(UnaryOpType.ELU, _packed);
      } else if (activation === 'relu6') {
        activationOpSnippet = getUnaryOpString(UnaryOpType.RELU6, _packed);
      } else if (activation === 'prelu') {
        activationOpSnippet = getBinaryOpString(BinaryOpType.PRELU, _packed);
      } else if (activation === 'sigmoid') {
        activationOpSnippet = getUnaryOpString(UnaryOpType.SIGMOID, _packed);
      } else if (activation === 'leakyrelu') {
        activationOpSnippet = getUnaryOpString(UnaryOpType.LEAKYRELU, _packed);
      } else {
        throw new Error(`Activation ${activation} has not been implemented for the WebGPU backend.`);
      }
      const elementSize = _packed ? 4 : 1;
      const dataType = typeSnippet(elementSize, 'f32');
      let activationFnSnippet = '';
      if (_hasPreluActivationWeights) {
        activationFnSnippet = `
    fn activation(a : ${dataType}, coords : vec${_coordsLength}<i32>) -> ${dataType} {
      let b = getPreluActivationWeightsByOutputCoords(coords);
      ${activationOpSnippet}
    }`;
      } else {
        activationFnSnippet = `
    fn activation(a : ${dataType}, coords : vec${_coordsLength}<i32>) -> ${dataType} {
      ${activationOpSnippet}
    }`;
      }
      return activationFnSnippet;
    };

export const biasActivationSnippet = (hasBias: boolean, activation?: Activation): string => `
      ${hasBias ? 'value = value + getBiasByOutputCoords(coords);' : ''}
      ${activation ? 'value = activation(value, coords);' : ''}
      `;
