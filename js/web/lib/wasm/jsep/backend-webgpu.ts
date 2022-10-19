// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {Tensor} from './tensor';
import {createGpuDataManager, GpuDataManager} from './webgpu/gpu-data-manager';
import {WEBGPU_OP_RESOLVE_RULES} from './webgpu/op-resolve-rules';
import {ProgramManager} from './webgpu/program-manager';
import {ComputeContext, GpuData, GpuDataType, ProgramInfo, ProgramInfoLoader} from './webgpu/types';

const getProgramInfoUniqueKey =
    (programInfo: ProgramInfo|ProgramInfoLoader, inputTensors: readonly Tensor[], inputGpuDatas: readonly GpuData[]):
        string => {
          const inputGpuDataTypes = inputGpuDatas.map(data => `${data.type}`).join('_');
          const inputTensorShapes = inputTensors.map(t => `${t.dims.join(',')}`).join('_');
          let key = programInfo.name;
          if (programInfo.cacheHint) {
            key += '[' + programInfo.cacheHint + ']';
          }
          key += ':' + inputTensorShapes + ';' + inputGpuDataTypes;
          return key;
        };

export class WebGpuBackend {
  device: GPUDevice;
  gpuDataManager: GpuDataManager;
  programManager: ProgramManager;

  kernelAttributes: Map<number, [(context: ComputeContext) => number, unknown]>;

  commandEncoder: GPUCommandEncoder|null = null;
  computePassEncoder: GPUComputePassEncoder|null = null;
  pendingDispatchNumber = 0;

  async initialize(): Promise<void> {
    if (!navigator.gpu) {
      // WebGPU is not available.
      throw new Error('WebGpuBackend: WebGPU is not available.');
    }

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      throw new Error('WebGpuBackend: Failed to get GPU adapter.');
    }
    this.device = await adapter.requestDevice();
    this.gpuDataManager = createGpuDataManager(this);
    this.programManager = new ProgramManager(this);
    this.kernelAttributes = new Map();
    // TODO: set up flags

    this.device.onuncapturederror = ev => {
      if (ev.error instanceof GPUValidationError) {
        // eslint-disable-next-line no-console
        console.error(`An uncaught WebGPU validation error was raised: ${ev.error.message}`);
      }
    };
  }

  dispose(): void {
    // TODO: uninitialization
    // this.glContext.dispose();
  }

  getCommandEncoder(): GPUCommandEncoder {
    if (!this.commandEncoder) {
      this.commandEncoder = this.device.createCommandEncoder();
    }
    return this.commandEncoder;
  }

  getComputePassEncoder(): GPUComputePassEncoder {
    if (!this.computePassEncoder) {
      this.computePassEncoder = this.getCommandEncoder().beginComputePass();
    }
    return this.computePassEncoder;
  }

  endComputePass(): void {
    if (this.computePassEncoder) {
      this.computePassEncoder.end();
      this.computePassEncoder = null;
    }
  }

  flush(): void {
    this.endComputePass();
    this.device.queue.submit([this.getCommandEncoder().finish()]);
    this.commandEncoder = null;
    this.pendingDispatchNumber = 0;
  }

  private uploadGpuData(tensor: Tensor, textureType: GpuDataType): GpuData {
    return this.gpuDataManager.upload(tensor, textureType);
  }

  private createGpuData(type: Tensor.DataType, dims: readonly number[], gpuDataType: GpuDataType): [Tensor, GpuData] {
    return this.dataManager.createGpuTensor(type, dims, gpuDataType);
  }

  run(program: ProgramInfoLoader|ProgramInfo, inputs: readonly Tensor[]): Tensor[] {
    if (inputs.length !== program.inputTypes.length) {
      throw new Error(`Input size must be equal to ${program.inputTypes.length}.`);
    }

    // create info for inputs
    const inputDatas: GpuData[] = [];
    for (let i = 0; i < program.inputTypes.length; ++i) {
      inputDatas[i] = this.uploadGpuData(inputs[i], program.inputTypes[i]);
    }

    const key = getProgramInfoUniqueKey(program, inputs, inputDatas);
    let artifact = this.programManager.getArtifact(key);
    const programInfo = artifact ?
        artifact.programInfo :
        (typeof (program as ProgramInfoLoader).get === 'function' ? (program as ProgramInfoLoader).get() :
                                                                    (program as ProgramInfo));

    // create info for outputs
    const outputDatas: GpuData[] = [];
    const outputTensors: Tensor[] = [];
    for (let i = 0; i < programInfo.outputs.length; ++i) {
      const [tensor, gpuData] = this.createGpuData(
          programInfo.outputs[i].type, programInfo.outputs[i].dims, programInfo.outputs[i].gpuDataType);
      outputTensors.push(tensor);
      outputDatas.push(gpuData);
    }

    if (!artifact) {
      artifact = this.programManager.build(programInfo);
      this.programManager.setArtifact(key, artifact);
    }

    this.programManager.run(artifact, inputDatas, outputDatas, artifact.programInfo.dispatchGroup(inputs));

    return outputTensors;
  }

  upload(gpuDataId: number, data: Uint8Array) {
    this.gpuDataManager.upload(gpuDataId, data);
  }

  async download(gpuDataId: number, data: Uint8Array) {
    const arrayBuffer = await this.gpuDataManager.download(gpuDataId);
    data.set(new Uint8Array(arrayBuffer));
  }

  alloc(size: number): number {
    return this.gpuDataManager.create(size).id;
  }

  free(ptr: number): number {
    return this.gpuDataManager.release(ptr);
  }

  createKernel(name: string, kernelId: number, attribute: unknown) {
    const lookup = WEBGPU_OP_RESOLVE_RULES.get(name);
    if (!lookup) {
      throw new Error(`kernel not implemented: ${name}`);
    }

    if (Array.isArray(lookup)) {
      const init = lookup[1];
    }
    this.kernelAttributes.set(kernelId)
  }

  releaseKernel(kernelId: number) {}

  computeKernel(kernelId: number, context: ComputeContext): number {
    throw new Error('Method not implemented.');
  }
}
