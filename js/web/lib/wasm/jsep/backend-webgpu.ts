// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {env} from 'onnxruntime-common';

import {TensorView} from './tensor';
import {createGpuDataManager, GpuDataManager} from './webgpu/gpu-data-manager';
import {RunFunction, WEBGPU_OP_RESOLVE_RULES} from './webgpu/op-resolve-rules';
import {ProgramManager} from './webgpu/program-manager';
import {ComputeContext, GpuData, GpuDataType, ProgramInfo, ProgramInfoLoader} from './webgpu/types';

const getProgramInfoUniqueKey =
    (programInfo: ProgramInfo|ProgramInfoLoader, inputTensorShapes: ReadonlyArray<TensorView['dims']>,
     inputGpuDataTypes: readonly GpuDataType[]): string => {
      const inputTensorShapesToString = inputTensorShapes.map(d => `${d.join(',')}`).join('_');
      const inputGpuDataTypesToString = inputGpuDataTypes.join('_');
      let key = programInfo.name;
      if (programInfo.cacheHint) {
        key += '[' + programInfo.cacheHint + ']';
      }
      key += ':' + inputTensorShapesToString + ';' + inputGpuDataTypesToString;
      return key;
    };

export class WebGpuBackend {
  device: GPUDevice;
  gpuDataManager: GpuDataManager;
  programManager: ProgramManager;

  temporaryData: GpuData[];

  // TODO: remove value[0]. the string is only for debug
  kernels: Map<number, [string, RunFunction, unknown]>;

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
    this.gpuDataManager.refreshPendingBuffers();
    this.commandEncoder = null;
    this.pendingDispatchNumber = 0;
  }

  run(program: ProgramInfoLoader|ProgramInfo, inputs: readonly TensorView[], outputIndices: readonly number[],
      createKernelOutput: (index: number, dataType: number, dims: readonly number[]) => TensorView,
      createTemporaryOutput: (dataType: number, dims: readonly number[]) => TensorView): TensorView[] {
    if (inputs.length !== program.inputTypes.length) {
      throw new Error(`Input size must be equal to ${program.inputTypes.length}.`);
    }

    // create info for inputs
    const inputDatas: GpuData[] = [];
    for (let i = 0; i < inputs.length; ++i) {
      const gpuData = this.gpuDataManager.get(inputs[i].data);
      if (!gpuData) {
        throw new Error(`no GPU data for input: ${inputs[i].data}`);
      }
      inputDatas[i] = gpuData;
    }

    const key = getProgramInfoUniqueKey(program, inputs.map(i => i.dims), inputDatas.map(i => i.type));
    let artifact = this.programManager.getArtifact(key);
    const programInfo = artifact ?
        artifact.programInfo :
        (typeof (program as ProgramInfoLoader).get === 'function' ? (program as ProgramInfoLoader).get() :
                                                                    (program as ProgramInfo));

    // check ouput indices
    const validatedOutputIndices = outputIndices.length === 0 ? programInfo.outputs.map((_, i) => i) : outputIndices;
    if (validatedOutputIndices.length !== programInfo.outputs.length) {
      throw new Error(`Output size must be equal to ${programInfo.outputs.length}.`);
    }

    // create info for outputs
    const outputTensorViews: TensorView[] = [];
    const outputDatas: GpuData[] = [];
    for (let i = 0; i < programInfo.outputs.length; ++i) {
      const tensorView = validatedOutputIndices[i] === -1 ?
          createTemporaryOutput(programInfo.outputs[i].dataType, programInfo.outputs[i].dims) :
          createKernelOutput(validatedOutputIndices[i], programInfo.outputs[i].dataType, programInfo.outputs[i].dims);
      const gpuData = this.gpuDataManager.get(tensorView.data);
      if (!gpuData) {
        throw new Error(`no GPU data for output: ${tensorView.data}`);
      }
      outputTensorViews.push(tensorView);
      outputDatas.push(gpuData);
    }

    if (!artifact) {
      artifact = this.programManager.build(programInfo);
      this.programManager.setArtifact(key, artifact);
    }

    this.programManager.run(artifact, inputDatas, outputDatas, artifact.programInfo.dispatchGroup(inputs));

    return outputTensorViews;
  }

  upload(gpuDataId: number, data: Uint8Array): void {
    this.gpuDataManager.upload(gpuDataId, data);
  }

  memcpy(src: number, dst: number): void {
    this.gpuDataManager.memcpy(src, dst);
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
    this.kernels.set(kernelId, [name, op[0], processedAttribute]);
  }

  releaseKernel(kernelId: number): void {
    this.kernels.delete(kernelId);
  }

  computeKernel(kernelId: number, context: ComputeContext): number {
    const kernel = this.kernels.get(kernelId);
    if (!kernel) {
      throw new Error(`kernel not created: ${kernelId}`);
    }
    const [name, kernelEntry, attributes] = kernel;

    if (env.debug) {
      // eslint-disable-next-line no-console
      console.log(`[js] Start to run kernel "${name}"...`);
    }

    this.temporaryData = [];
    try {
      return kernelEntry(context, attributes);
    } finally {
      for (const data of this.temporaryData) {
        this.gpuDataManager.release(data.id);
      }
      this.temporaryData = [];
    }
  }
}
