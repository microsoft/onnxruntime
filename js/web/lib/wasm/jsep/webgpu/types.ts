// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {Tensor} from '../tensor';

export enum GpuDataType {
  default = 0,
  upload = 1,
  profile = 2
}
export type GpuDataId = number;

export interface GpuData {
  type: GpuDataType;
  id: GpuDataId;
  buffer: GPUBuffer;
}

export interface TensorInfo {
  id?: Tensor.Id;
  dims: readonly number[];
  type: Tensor.DataType;
  gpuDataType: GpuDataType;
}


export interface ProgramVariable {
  type: 'float'|'int';
  name: string;
  arrayLength?: number;
  data: number|number[];
}


export interface ProgramMetadata {
  /**
   * the name of the program. used for debugging and profiling
   */
  name: string;

  // inputLayouts: GPUBindGroupLayoutEntry[];
  // outputLayouts: GPUBindGroupLayoutEntry[];

  /**
   * gpu data types for each input
   */
  inputTypes: GpuDataType[];
  /**
   * an optional string as a cache hint in the artifact cache
   */
  cacheHint?: string;
}

/**
 * A ProgramInfoLoader allows
 */
export interface ProgramInfoLoader extends ProgramMetadata {
  /**
   * a function to get the program info
   */
  get(): ProgramInfo;
}

/**
 * A set of data that represent a shader program
 */
export interface ProgramInfo extends ProgramMetadata {
  /**
   * information of uniform variables
   */
  variables?: ProgramVariable[];
  /**
   * tensor info for outputs
   */
  outputs: TensorInfo[];
  /**
   * the shader's processing source code
   */
  shaderSource: string;
  /**
   * default is "main"
   */
  // entryPoint: string;

  dispatchGroup: (inputs: readonly Tensor[]) => {
    x: number;
    y?: number;
    z?: number;
  };
}

export interface Artifact {
  programInfo: ProgramInfo;
  computePipeline: GPUComputePipeline;
  // attribLocations: {position: number; textureCoord: number};
}
