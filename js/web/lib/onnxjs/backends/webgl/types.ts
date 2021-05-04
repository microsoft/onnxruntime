// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {Tensor} from '../../tensor';

import {WebGLInferenceHandler} from './inference-handler';
import {WebGLContext} from './webgl-context';

/**
 * Represent an operator instance that can run in WebGL backend
 */
export interface WebGLOperator {
  createProgramInfo(handler: WebGLInferenceHandler, inputs: Tensor[]): ProgramInfo;
  createRunData(handler: WebGLInferenceHandler, programInfo: ProgramInfo, inputs: Tensor[]): RunData;
}

/**
 * Layout info is used for mapping n-dimensional array to 2D textures
 * The layout is created by the TextureLayoutStrategy based on
 * the Tensor's dimensions and strides
 */
export interface TextureLayout {
  width: number;
  height: number;
  /**
   * specify the number of value that encoded in a single pixel
   */
  channels: 1|2|3|4;
  /**
   * whether in packed mode or not
   */
  isPacked?: boolean;
  /**
   * the normalized shape
   */
  shape: readonly number[];
  /**
   * the stride of each dimensions, calculated according to shape
   */
  strides: readonly number[];
  /**
   * the original shape(dims) of the corresponding tensor
   */
  unpackedShape: readonly number[];

  reversedWH?: boolean;
}
export interface TextureData extends TextureLayout {
  tensor: Tensor;
  texture: WebGLTexture;
}

/**
 * A set of data that represent a shader program
 */
export interface ProgramInfo {
  /**
   * texture layouts for each input
   */
  inputLayouts: TextureLayout[];
  /**
   * names of uniform samplers
   */
  samplers: string[];
  /**
   * information of uniform variables
   */
  variables?: VariableInfo[];
  /**
   * texture layout for output
   */
  outputLayout: TextureLayout;
  /**
   * the shader's processing source code
   */
  shaderSource: string;
  /**
   * whether the shader source contains a customized main function implementation
   */
  hasMain?: boolean;
  params?: {[name: string]: number|number[]|string};

  expectPackedInputs?: boolean;
  expectPackedOutputs?: boolean;
}

export interface VariableInfo {
  type: 'float'|'int';
  name: string;
  arrayLength?: number;
}

/**
 * Information of uniforms that shader uses
 */
export interface UniformInfo {
  type: 'sampler2D'|VariableInfo['type'];
  name: string;
  arrayLength?: number;
}

export interface UniformLocation extends UniformInfo {
  location: WebGLUniformLocation;
}

/**
 * Artifact is the result of compilation
 * It does not contain input of output data
 * However anything that could be run as a "program"
 */
export interface Artifact {
  programInfo: ProgramInfo;
  program: WebGLProgram;
  uniformLocations: UniformLocation[];
  attribLocations: {position: number; textureCoord: number};
}
export declare namespace Artifact {
  type UniformLocations = Artifact['uniformLocations'];
  type AttribLocations = Artifact['attribLocations'];
}

export interface UniformData {
  [name: string]: number|number[];
}

/**
 * RunData contains all inputs that required to run a "program"
 */
export interface RunData {
  inputTextureDatas: TextureData[];
  outputTextureData: TextureData;
  uniformData: UniformData;
  draw?: (glContext: WebGLContext, artifact: Artifact) => void;
}
