// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {Env, Tensor} from 'onnxruntime-common';

import {configureLogger, LOG_DEBUG} from './log';
import {createView, TensorView} from './tensor-view';
import {createGpuDataManager, downloadGpuData, GpuDataManager} from './webgpu/gpu-data-manager';
import {RunFunction, WEBGPU_OP_RESOLVE_RULES} from './webgpu/op-resolve-rules';
import {ProgramManager} from './webgpu/program-manager';
import {ComputeContext, GpuData, ProgramInfo, ProgramInputTensorInfoDependency} from './webgpu/types';

const getProgramInputTensorInfoDependencyKey =
    (inputTensors: readonly TensorView[], inputDependencies: readonly ProgramInputTensorInfoDependency[]): string => {
      if (inputDependencies.length !== inputTensors.length) {
        throw new Error(`inputDependencies length ${inputDependencies.length} is not equal to inputTensors length ${
            inputTensors.length}.`);
      }

      const inputInfos: string[] = [];
      for (let i = 0; i < inputTensors.length; ++i) {
        const type = inputTensors[i].dataType;
        switch (inputDependencies[i]) {
          case 'none': {
            inputInfos.push('');
            break;
          }
          case 'type': {
            inputInfos.push(`${type}`);
            break;
          }
          case 'rank': {
            const rank = inputTensors[i].dims.length;
            inputInfos.push(`${type};${rank}`);
            break;
          }
          case 'dims': {
            const dims = inputTensors[i].dims.join(',');
            inputInfos.push(`${type};${dims}`);
            break;
          }
          default:
            throw new Error(`unsupported input dependency: ${inputDependencies[i]}`);
        }
      }

      return inputInfos.join('|');
    };

/**
 * get a unique key representing the program from the program info, input shapes and types.
 *
 * @returns a unique key is a shorter string than the shader source, which contains all the information to identify a
 * program. if the key is the same, the program shader source should be the same, so we can reuse the program.
 *
 */
const getProgramInfoUniqueKey =
    (programInfo: ProgramInfo, inputTensors: readonly TensorView[], is1DimensionDispatch: boolean): string => {
      // final key format:
      // <PROGRAM_NAME>[<PROGRAM_CUSTOM_CACHE_HINT>]:is1DimensionDispatch:<INPUTS_INFO_0>|<INPUTS_INFO_1>|...
      let key = programInfo.name;
      if (programInfo.shaderCache?.hint) {
        key += '[' + programInfo.shaderCache.hint + ']';
      }
      key += ':' + is1DimensionDispatch +
          `:${
                 getProgramInputTensorInfoDependencyKey(
                     inputTensors,
                     programInfo.shaderCache?.inputDependencies ??
                         new Array<ProgramInputTensorInfoDependency>(inputTensors.length).fill('dims'))}`;
      return key;
    };

/**
 * this class is designed to store status and being used as a singleton for JSEP. It will be passed to jsepInit() as
 * the first parameter so that it is stored for future use.
 */
export class WebGpuBackend {
  device: GPUDevice;
  /**
   * an instance of GpuDataManager to manage a GpuDataId -> GpuBuffer mapping
   */
  gpuDataManager: GpuDataManager;
  /**
   * an instance of ProgramManager to build and run WebGPU compute shader program, and manage a ProgramKey -> Program
   * artifacts mapping
   */
  programManager: ProgramManager;

  /**
   * representing the kernel ID of which is currently being computed (CPU code perspective).
   * `null` means no kernel is being computed.
   * only one kernel can be computed at a moment.
   */
  currentKernelId: number|null = null;
  /**
   * a list of temporary GPU data for the current kernel. should release when the kernel done computation.
   */
  private temporaryData: GpuData[];
  /**
   * a KernelID -> a GPU data list, which stores persistent GPU data owned by the specific kernel.
   */
  private kernelPersistentData: Map<number, GpuData[]>;
  /**
   * a KernelID -> a custom data, which stores custom data owned by the specific kernel.
   */
  private kernelCustomData: Map<number, {[key: string]: unknown}>;
  /**
   * get the custom data of the current kernel
   */
  get currentKernelCustomData(): {[key: string]: unknown} {
    if (this.currentKernelId === null) {
      throw new Error('currentKernelCustomData(): currentKernelId is null. (should not happen)');
    }

    let data = this.kernelCustomData.get(this.currentKernelId);
    if (!data) {
      data = {};
      this.kernelCustomData.set(this.currentKernelId, data);
    }

    return data;
  }

  /**
   * a KernelID -> kernel info mapping. value is
   * [ op_type, name, run function, [optional] preprocess_attribute_once function ]
   */
  kernels: Map<number, [string, string, RunFunction, [((attribute: unknown) => unknown) | undefined, unknown]]>;

  private commandEncoder: GPUCommandEncoder|null = null;
  private computePassEncoder: GPUComputePassEncoder|null = null;
  pendingDispatchNumber = 0;

  queryData?: GpuData;
  querySet?: GPUQuerySet;
  querySetCount = 2;
  queryTimeBase?: bigint;

  env: Env;

  /**
   * a SessionID -> a Map of (InputOutputIndex -> [ID, GPUBuffer]) mapping.
   */
  sessionExternalDataMapping: Map<number, Map<number, [number, GPUBuffer]>> = new Map();

  async initialize(env: Env, adapter: GPUAdapter): Promise<void> {
    this.env = env;
    const requiredFeatures: GPUFeatureName[] = [];
    const deviceDescriptor: GPUDeviceDescriptor = {
      requiredLimits: {
        maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
        maxComputeWorkgroupsPerDimension: adapter.limits.maxComputeWorkgroupsPerDimension,
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxBufferSize: adapter.limits.maxBufferSize,
        maxComputeInvocationsPerWorkgroup: adapter.limits.maxComputeInvocationsPerWorkgroup,
        maxComputeWorkgroupSizeX: adapter.limits.maxComputeWorkgroupSizeX,
        maxComputeWorkgroupSizeY: adapter.limits.maxComputeWorkgroupSizeY,
        maxComputeWorkgroupSizeZ: adapter.limits.maxComputeWorkgroupSizeZ,
      },
      requiredFeatures,
    };

    if (adapter.features.has('timestamp-query')) {
      requiredFeatures.push('timestamp-query');
    }
    if (adapter.features.has('shader-f16')) {
      requiredFeatures.push('shader-f16');
    }

    this.device = await adapter.requestDevice(deviceDescriptor);
    this.gpuDataManager = createGpuDataManager(this);
    this.programManager = new ProgramManager(this);
    this.kernels = new Map();
    this.kernelPersistentData = new Map();
    this.kernelCustomData = new Map();

    // set up flags for logger
    configureLogger(env.logLevel!, !!env.debug);

    // TODO: set up flags

    this.device.onuncapturederror = ev => {
      if (ev.error instanceof GPUValidationError) {
        // eslint-disable-next-line no-console
        console.error(`An uncaught WebGPU validation error was raised: ${ev.error.message}`);
      }
    };

    Object.defineProperty(this.env.webgpu, 'device', {value: this.device});
  }

  dispose(): void {
    if (typeof this.querySet !== 'undefined') {
      this.querySet.destroy();
    }
    this.gpuDataManager.dispose();
  }

  getCommandEncoder(): GPUCommandEncoder {
    if (!this.commandEncoder) {
      this.commandEncoder = this.device.createCommandEncoder();
    }
    return this.commandEncoder;
  }

  getComputePassEncoder(): GPUComputePassEncoder {
    if (!this.computePassEncoder) {
      const computePassDescriptor: GPUComputePassDescriptor = {};
      if (this.isQueryEnabled()) {
        if (typeof this.querySet === 'undefined') {
          this.querySet = this.device.createQuerySet({
            type: 'timestamp',
            count: this.querySetCount,
          });
        }
        computePassDescriptor.timestampWrites = {
          querySet: this.querySet,
          beginningOfPassWriteIndex: 0,
          endOfPassWriteIndex: 1,
        };
      }

      this.computePassEncoder = this.getCommandEncoder().beginComputePass(computePassDescriptor);
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
    if (this.commandEncoder) {
      this.endComputePass();
      this.device.queue.submit([this.getCommandEncoder().finish()]);
      this.gpuDataManager.refreshPendingBuffers();
      this.commandEncoder = null;
      this.pendingDispatchNumber = 0;
    }
  }

  isQueryEnabled(): boolean {
    if (this.device.features.has('timestamp-query') && this.env.webgpu.profilingMode === 'default') {
      return true;
    } else {
      return false;
    }
  }

  /**
   * run a WebGPU program.
   * @param program a ProgramInfo instance
   * @param inputTensorViews a TensorView array. each element represents a value already exists in GPU.
   * @param outputIndices an indices array. each element can be either -1 (temporary data), -2 (persistent data) or an
   * index to the kernel's output.
   * @param createKernelOutput a callback function that create a value to kernel's output with the given index
   * @param createIntermediateOutput a callback function that create a value as a intermediate value, either temporary
   * or persistent (owned by the current kernel)
   * @returns a TensorView array representing the result.
   */
  run(program: ProgramInfo, inputTensorViews: readonly TensorView[], outputIndices: readonly number[],
      createKernelOutput: (index: number, dataType: number, dims: readonly number[]) => TensorView,
      createIntermediateOutput: (dataType: number, dims: readonly number[]) => TensorView): TensorView[] {
    // create info for inputs
    const inputDatas: GpuData[] = [];
    for (let i = 0; i < inputTensorViews.length; ++i) {
      const gpuData = this.gpuDataManager.get(inputTensorViews[i].data);
      if (!gpuData) {
        throw new Error(`no GPU data for input: ${inputTensorViews[i].data}`);
      }
      inputDatas[i] = gpuData;
    }

    const {outputs, dispatchGroup, programUniforms} = program.getRunData(inputTensorViews);

    // check output indices
    const validatedOutputIndices = outputIndices.length === 0 ? outputs.map((_, i) => i) : outputIndices;
    if (validatedOutputIndices.length !== outputs.length) {
      throw new Error(`Output size ${validatedOutputIndices.length} must be equal to ${outputs.length}.`);
    }

    // create info for outputs
    const outputTensorViews: TensorView[] = [];
    const outputDatas: GpuData[] = [];
    for (let i = 0; i < outputs.length; ++i) {
      // value -1 and -2 are used for creating temporary and persistent outputs.
      // value -3 is used for placeholder output. So -3, -2, -1 and 0, 1, 2, ... are valid
      // output indices. see type definition of ComputeContextInputsOutputsMapping for more details.
      if (!Number.isInteger(validatedOutputIndices[i]) || validatedOutputIndices[i] < -3 ||
          validatedOutputIndices[i] >= outputs.length) {
        throw new Error(`Invalid output index: ${validatedOutputIndices[i]}`);
      }
      if (validatedOutputIndices[i] === -3) {
        continue;
      }
      const isTemporary = validatedOutputIndices[i] === -1;
      const isPersistent = validatedOutputIndices[i] === -2;
      const tensorView = (isTemporary || isPersistent) ?
          createIntermediateOutput(outputs[i].dataType, outputs[i].dims) :
          createKernelOutput(validatedOutputIndices[i], outputs[i].dataType, outputs[i].dims);
      const gpuData = this.gpuDataManager.get(tensorView.data);
      if (!gpuData) {
        throw new Error(`no GPU data for output: ${tensorView.data}`);
      }
      if (isTemporary) {
        this.temporaryData.push(gpuData);
      }
      if (isPersistent) {
        let persistentData = this.kernelPersistentData.get(this.currentKernelId!);
        if (!persistentData) {
          persistentData = [];
          this.kernelPersistentData.set(this.currentKernelId!, persistentData);
        }
        persistentData.push(gpuData);
      }
      outputTensorViews.push(tensorView);
      outputDatas.push(gpuData);
    }


    // load uniforms
    // TODO: add cache for uniform (is it necessary?)
    //
    let uniformBufferBinding: GPUBindingResource|undefined;
    if (programUniforms) {
      let currentOffset = 0;
      const offsets: number[] = [];

      programUniforms.forEach(v => {
        const data = typeof v.data === 'number' ? [v.data] : v.data;
        if (data.length === 0) {
          return;
        }
        // https://www.w3.org/TR/WGSL/#alignof
        const baseAlignment = data.length <= 2 ? data.length * 4 : 16;
        currentOffset = Math.ceil(currentOffset / baseAlignment) * baseAlignment;
        offsets.push(currentOffset);
        // When data.length > 4, the uniform variable is of type array<vec4<i32|u32|f32>,N>, where N =
        // Math.ceil(data.length / 4) and SizeOf(vec4<i32|u32|f32>) = 16. The total byte length is N *
        // SizeOf(vec4<i32|u32|f32>).
        currentOffset += data.length > 4 ? Math.ceil(data.length / 4) * 16 : data.length * 4;
      });

      // Meet alignment of struct here: https://www.w3.org/TR/WGSL/#alignment-and-size. For simplicity, set
      // maxAlignmentOfField to 16 since the underlying buffer has been rounded up to 16.
      const maxAlignmentOfField = 16;
      currentOffset = Math.ceil(currentOffset / maxAlignmentOfField) * maxAlignmentOfField;
      const arrayBuffer = new ArrayBuffer(currentOffset);
      programUniforms.forEach((v, i) => {
        const offset = offsets[i];
        const data = typeof v.data === 'number' ? [v.data] : v.data;
        if (v.type === 'int32') {
          new Int32Array(arrayBuffer, offset, data.length).set(data);
        } else if (v.type === 'uint32') {
          new Uint32Array(arrayBuffer, offset, data.length).set(data);
        } else {
          new Float32Array(arrayBuffer, offset, data.length).set(data);
        }
      });

      const uniformBufferData =
          // eslint-disable-next-line no-bitwise
          this.gpuDataManager.create(currentOffset, GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM);
      this.device.queue.writeBuffer(uniformBufferData.buffer, 0, arrayBuffer, 0, currentOffset);
      this.gpuDataManager.release(uniformBufferData.id);
      uniformBufferBinding = {offset: 0, size: currentOffset, buffer: uniformBufferData.buffer};
    }

    const normalizedDispatchGroup = this.programManager.normalizeDispatchGroupSize(dispatchGroup);
    const is1DimensionDispatch = normalizedDispatchGroup[1] === 1 && normalizedDispatchGroup[2] === 1;
    // get program info
    const key = getProgramInfoUniqueKey(program, inputTensorViews, is1DimensionDispatch);
    let artifact = this.programManager.getArtifact(key);
    if (!artifact) {
      artifact = this.programManager.build(program, normalizedDispatchGroup);
      this.programManager.setArtifact(key, artifact);
      LOG_DEBUG('info', () => `[artifact] key: ${key}, programName: ${program.name}`);
    }

    LOG_DEBUG(
        'info',
        () => `[ProgramManager] run "${program.name}" (key=${key}) with ${normalizedDispatchGroup[0]}x${
            normalizedDispatchGroup[1]}x${normalizedDispatchGroup[2]}`);
    this.programManager.run(
        artifact, inputTensorViews, outputTensorViews, inputDatas, outputDatas, normalizedDispatchGroup,
        uniformBufferBinding);

    return outputTensorViews;
  }

  upload(gpuDataId: number, data: Uint8Array): void {
    this.gpuDataManager.upload(gpuDataId, data);
  }

  memcpy(src: number, dst: number): void {
    this.gpuDataManager.memcpy(src, dst);
  }

  async download(gpuDataId: number, getTargetBuffer: () => Uint8Array): Promise<void> {
    // the underlying buffer may be changed after the async function is called. so we use a getter function to make sure
    // the buffer is up-to-date.
    await this.gpuDataManager.download(gpuDataId, getTargetBuffer);
  }

  alloc(size: number): number {
    return this.gpuDataManager.create(size).id;
  }

  free(ptr: number): number {
    return this.gpuDataManager.release(ptr);
  }

  createKernel(opType: string, kernelId: number, attribute: unknown, nodeName: string): void {
    const op = WEBGPU_OP_RESOLVE_RULES.get(opType);
    if (!op) {
      throw new Error(`kernel not implemented: ${opType}`);
    }

    this.kernels.set(kernelId, [opType, nodeName, op[0], [op[1], attribute]]);
  }

  releaseKernel(kernelId: number): void {
    const persistentData = this.kernelPersistentData.get(kernelId);
    if (persistentData) {
      for (const data of persistentData) {
        this.gpuDataManager.release(data.id);
      }
      this.kernelPersistentData.delete(kernelId);
    }

    this.kernelCustomData.delete(kernelId);
    this.kernels.delete(kernelId);
  }

  computeKernel(kernelId: number, context: ComputeContext, errors: Array<Promise<string|null>>): number {
    const kernel = this.kernels.get(kernelId);
    if (!kernel) {
      throw new Error(`kernel not created: ${kernelId}`);
    }
    const [opType, nodeName, kernelEntry, attributes] = kernel;
    if (this.currentKernelId !== null) {
      throw new Error(`kernel "[${opType}] ${nodeName}" is not allowed to be called recursively`);
    }
    this.currentKernelId = kernelId;

    // parse attributes if necessary
    if (attributes[0]) {
      attributes[1] = attributes[0](attributes[1]);
      attributes[0] = undefined;
    }

    LOG_DEBUG('info', () => `[WebGPU] Start to run kernel "[${opType}] ${nodeName}"...`);

    const useErrorScope = this.env.debug;

    this.temporaryData = [];
    try {
      if (useErrorScope) {
        this.device.pushErrorScope('validation');
      }

      kernelEntry(context, attributes[1]);
      return 0;  // ORT_OK
    } catch (e) {
      errors.push(Promise.resolve(`[WebGPU] Kernel "[${opType}] ${nodeName}" failed. ${e}`));
      return 1;  // ORT_FAIL
    } finally {
      if (useErrorScope) {
        errors.push(this.device.popErrorScope().then(
            err => err ? `GPU validation error for kernel "[${opType}] ${nodeName}": ${err.message}` : null));
      }

      for (const data of this.temporaryData) {
        this.gpuDataManager.release(data.id);
      }
      this.temporaryData = [];
      this.currentKernelId = null;
    }
  }

  // #region external buffer
  registerBuffer(sessionId: number, index: number, buffer: GPUBuffer, size: number): number {
    let sessionInputOutputMapping = this.sessionExternalDataMapping.get(sessionId);
    if (!sessionInputOutputMapping) {
      sessionInputOutputMapping = new Map();
      this.sessionExternalDataMapping.set(sessionId, sessionInputOutputMapping);
    }

    const previousBuffer = sessionInputOutputMapping.get(index);
    const id = this.gpuDataManager.registerExternalBuffer(buffer, size, previousBuffer?.[1]);
    sessionInputOutputMapping.set(index, [id, buffer]);
    return id;
  }
  unregisterBuffers(sessionId: number): void {
    const sessionInputOutputMapping = this.sessionExternalDataMapping.get(sessionId);
    if (sessionInputOutputMapping) {
      sessionInputOutputMapping.forEach(bufferInfo => this.gpuDataManager.unregisterExternalBuffer(bufferInfo[1]));
      this.sessionExternalDataMapping.delete(sessionId);
    }
  }
  getBuffer(gpuDataId: number): GPUBuffer {
    const gpuData = this.gpuDataManager.get(gpuDataId);
    if (!gpuData) {
      throw new Error(`no GPU data for buffer: ${gpuDataId}`);
    }
    return gpuData.buffer;
  }
  createDownloader(gpuBuffer: GPUBuffer, size: number, type: Tensor.GpuBufferDataTypes):
      () => Promise<Tensor.DataType> {
    return async () => {
      const data = await downloadGpuData(this, gpuBuffer, size);
      return createView(data.buffer, type);
    };
  }
  // #endregion
}
