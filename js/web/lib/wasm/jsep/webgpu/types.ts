// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {Tensor, TensorView} from '../tensor';

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
  dataType: number;
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

  dispatchGroup: (inputs: readonly TensorView[]) => {
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

export interface ComputeContextInputsOutputsMapping {
  /**
   * specify the mapping to the program's inputs. the value can be a number or a tensor view.
   * - if it's a number, it's the index of the kernel's input
   * - if it's a tensor view, it's an existing tensor view that will be used as the input
   *
   * if inputs is not specified, the mapping will be the kernel's inputs in order.
   */
  readonly inputs?: ReadonlyArray<TensorView|number>;
  /**
   * specify the mapping to the program's outputs. the value can be a number or undefined.
   * - if it's a non-negative number, it's the index of the kernel's output
   * - if it's -1, it's an output that will be created as a temporary value. this value will be released after
   * the kernel is executed.
   *
   * if outputs is not specified, the mapping will be the kernel's outputs in order.
   */
  readonly outputs?: readonly number[];
}

export interface ComputeContext {
  readonly opKernelContext: number;
  readonly inputs: readonly TensorView[];
  compute(program: ProgramInfoLoader|ProgramInfo, inputsOutputsMapping?: ComputeContextInputsOutputsMapping):
      TensorView[];
  output(index: number, dims: readonly number[]): number;
}
