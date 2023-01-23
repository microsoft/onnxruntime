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

export declare type Activation = 'linear' | 'relu' | 'prelu' | 'elu' | 'relu6' | 'leakyrelu' | 'sigmoid';

export const typeSnippet = (component: number) => {
  switch (component) {
    case 1:
      return 'f32';
    case 2:
      return 'vec2<f32>';
    case 3:
      return 'vec3<f32>';
    case 4:
      return 'vec4<f32>';
    default:
      throw new Error(`${component}-component is not supported.`);
  }
};

export const activationFnSnippet =
    (activation?: Activation, _hasPreluActivationWeights = false, _packed = false, _coordsLength = 3): string => {
      if (!activation) {
        return '';
      }

      return '';
      // let activationOpSnippet = '';
      // if (activation === 'linear') {
      //   activationOpSnippet = getUnaryOpString(UnaryOpType.LINEAR);
      // } else if (activation === 'relu') {
      //   activationOpSnippet = getUnaryOpString(UnaryOpType.RELU, packed);
      // } else if (activation === 'elu') {
      //   activationOpSnippet = getUnaryOpString(UnaryOpType.ELU, packed);
      // } else if (activation === 'relu6') {
      //   activationOpSnippet = getUnaryOpString(UnaryOpType.RELU6, packed);
      // } else if (activation === 'prelu') {
      //   activationOpSnippet = getBinaryOpString(BinaryOpType.PRELU, packed);
      // } else if (activation === 'sigmoid') {
      //   activationOpSnippet = getUnaryOpString(UnaryOpType.SIGMOID, packed);
      // } else if (activation === 'leakyrelu') {
      //   activationOpSnippet = getUnaryOpString(UnaryOpType.LEAKYRELU, packed);
      // } else {
      //   throw new Error(`Activation ${activation} has not been implemented for the WebGPU backend.`);
      // }
      // const elementSize = packed ? 4 : 1;
      // const dataType = typeSnippet(elementSize);
      // let activationFnSnippet = '';
      // if (hasPreluActivationWeights) {
      //   activationFnSnippet = `
      // fn activation(a : ${dataType}, coords : vec${coordsLength}<i32>) -> ${dataType} {
      //   let b = getPreluActivationWeightsByOutputCoords(coords);
      //   ${activationOpSnippet}
      // }`;
      // } else {
      //   activationFnSnippet = `
      // fn activation(a : ${dataType}, coords : vec${coordsLength}<i32>) -> ${dataType} {
      //   ${activationOpSnippet}
      // }`;
      // }
      // return activationFnSnippet;
    };

export const biasActivationSnippet = (hasBias: boolean, activation?: Activation): string => `
      ${hasBias ? 'value = value + getBiasByOutputCoords(coords);' : ''}
      ${activation ? 'value = activation(value, coords);' : ''}
      `;
