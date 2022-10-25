// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {TensorView} from './tensor';
import {createGpuDataManager, GpuDataManager} from './webgpu/gpu-data-manager';
import {RunFunction, WEBGPU_OP_RESOLVE_RULES} from './webgpu/op-resolve-rules';
import {ProgramManager} from './webgpu/program-manager';
import {ComputeContext, GpuData, ProgramInfo, ProgramInfoLoader} from './webgpu/types';

const getProgramInfoUniqueKey =
    (programInfo: ProgramInfo|ProgramInfoLoader, inputTensors: readonly TensorView[],
     inputGpuDatas: readonly GpuData[]): string => {
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

  kernels: Map<number, [RunFunction, unknown]>;

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
    this.kernels = new Map();
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

  run(program: ProgramInfoLoader|ProgramInfo, inputs: readonly TensorView[],
      createOutput: (index: number, dims: readonly number[]) => number): number {
    if (inputs.length !== program.inputTypes.length) {
      throw new Error(`Input size must be equal to ${program.inputTypes.length}.`);
    }

    // // create info for inputs
    // const inputDatas: GpuData[] = [];
    // for (let i = 0; i < program.inputTypes.length; ++i) {
    //   inputDatas[i] = this.uploadGpuData(inputs[i], program.inputTypes[i]);
    // }

    const inputDatas: GpuData[] = [];
    for (let i = 0; i < inputs.length; ++i) {
      const gpuData = this.gpuDataManager.get(inputs[i].data);
      if (!gpuData) {
        throw new Error(`no GPU data for ${inputs[i].data}`);
      }
      inputDatas[i] = gpuData;
    }

    const key = getProgramInfoUniqueKey(program, inputs, inputDatas);
    let artifact = this.programManager.getArtifact(key);
    const programInfo = artifact ?
        artifact.programInfo :
        (typeof (program as ProgramInfoLoader).get === 'function' ? (program as ProgramInfoLoader).get() :
                                                                    (program as ProgramInfo));

    // create info for outputs
    const outputDatas: GpuData[] = [];
    for (let i = 0; i < programInfo.outputs.length; ++i) {
      const dataId = createOutput(i, programInfo.outputs[i].dims);
      const gpuData = this.gpuDataManager.get(dataId);
      if (!gpuData) {
        throw new Error(`no GPU data for ${inputs[i].data}`);
      }
      outputDatas.push(gpuData);
    }

    if (!artifact) {
      artifact = this.programManager.build(programInfo);
      this.programManager.setArtifact(key, artifact);
    }

    this.programManager.run(artifact, inputDatas, outputDatas, artifact.programInfo.dispatchGroup(inputs));

    return 0;
  }

  upload(gpuDataId: number, data: Uint8Array): void {
    this.gpuDataManager.upload(gpuDataId, data);
  }

  async download(gpuDataId: number, data: Uint8Array): Promise<void> {
    const arrayBuffer = await this.gpuDataManager.download(gpuDataId);
    data.set(new Uint8Array(arrayBuffer));
  }

  alloc(size: number): number {
    return this.gpuDataManager.create(size).id;
  }

  free(ptr: number): number {
    return this.gpuDataManager.release(ptr);
  }

  createKernel(name: string, kernelId: number, attribute: unknown): void {
    const op = WEBGPU_OP_RESOLVE_RULES.get(name);
    if (!op) {
      throw new Error(`kernel not implemented: ${name}`);
    }

    let processedAttribute = attribute;
    if (op.length > 1 && typeof op[1] !== 'undefined') {
      processedAttribute = op[1](attribute);
    }
    this.kernels.set(kernelId, [op[0], processedAttribute]);
  }

  releaseKernel(kernelId: number): void {
    this.kernels.delete(kernelId);
  }

  computeKernel(kernelId: number, context: ComputeContext): number {
    const kernel = this.kernels.get(kernelId);
    if (!kernel) {
      throw new Error(`kernel not created: ${kernelId}`);
    }
    const [kernelEntry, attributes] = kernel;
    return kernelEntry(context, attributes);
  }
}
