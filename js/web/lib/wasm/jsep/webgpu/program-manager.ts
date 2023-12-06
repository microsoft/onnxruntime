// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {tensorDataTypeEnumToString} from '../../wasm-common';
import {WebGpuBackend} from '../backend-webgpu';
import {LOG_DEBUG} from '../log';
import {TensorView} from '../tensor-view';

import {createShaderHelper} from './ops/common';
import {Artifact, GpuData, ProgramInfo} from './types';

/**
 * ProgramManager is the main class behind running computations
 * It builds ProgramInfo's into Artifacts
 * It compiles given ProgramInfo's into WebGL Prorams (cached as Artifacts)
 * Uses the artifact to run the computation by calling Draw on
 * the WebGL drawing buffer
 * ProgramManager automatically maps (binds) input variables to their
 * corresponding Location's in the binary program
 */
export class ProgramManager {
  repo: Map<unknown, Artifact>;  // this should be per-session object
  attributesBound: boolean;

  constructor(private backend: WebGpuBackend) {
    this.repo = new Map();
    this.attributesBound = false;
  }
  getArtifact(key: unknown): Artifact|undefined {
    return this.repo.get(key);
  }
  setArtifact(key: unknown, artifact: Artifact): void {
    this.repo.set(key, artifact);
  }
  run(buildArtifact: Artifact, inputTensorViews: readonly TensorView[], outputTensorViews: readonly TensorView[],
      inputs: GpuData[], outputs: GpuData[], dispatchGroup: [number, number, number],
      uniformBufferBinding: GPUBindingResource|undefined): void {
    const device = this.backend.device;

    const computePassEncoder = this.backend.getComputePassEncoder();
    const webgpuEnv = this.backend.env.webgpu;
    const profilingEnabled = this.backend.supportTimestampQuery &&
        (webgpuEnv.profiling?.mode === 'default' ||
         (!webgpuEnv.profiling?.mode && webgpuEnv.profilingMode === 'default'));
    if (profilingEnabled) {
      // profiling write start timestamp

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (computePassEncoder as any).writeTimestamp(this.backend.profilingQuerySet, 0);
    }

    computePassEncoder.setPipeline(buildArtifact.computePipeline);
    const entries = [];
    for (const input of inputs) {
      entries.push({binding: entries.length, resource: {buffer: input.buffer}});
    }
    for (const output of outputs) {
      entries.push({binding: entries.length, resource: {buffer: output.buffer}});
    }
    if (uniformBufferBinding) {
      entries.push({binding: entries.length, resource: uniformBufferBinding});
    }
    const bindGroup = device.createBindGroup(
        {layout: buildArtifact.computePipeline.getBindGroupLayout(0), entries, label: buildArtifact.programInfo.name});
    computePassEncoder.setBindGroup(0, bindGroup);

    computePassEncoder.dispatchWorkgroups(...dispatchGroup);

    this.backend.pendingDispatchNumber++;

    if (this.backend.isQueryEnabled()) {
      if (typeof this.backend.queryData === 'undefined') {
        this.backend.queryData = this.backend.gpuDataManager.create(
            // eslint-disable-next-line no-bitwise
            this.backend.querySetCount * 8, GPUBufferUsage.COPY_SRC | GPUBufferUsage.QUERY_RESOLVE);
      }
      const syncData = this.backend.gpuDataManager.create(
          // eslint-disable-next-line no-bitwise
          this.backend.querySetCount * 8, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);

      this.backend.endComputePass();
      this.backend.getCommandEncoder().resolveQuerySet(this.backend.querySet!, 0, 2, this.backend.queryData.buffer, 0);
      this.backend.getCommandEncoder().copyBufferToBuffer(
          this.backend.queryData.buffer, 0, syncData.buffer, 0, this.backend.querySetCount * 8);
      this.backend.flush();

      const kernelId = this.backend.currentKernelId!;
      const kernelInfo = this.backend.kernels.get(kernelId)!;

      void syncData.buffer.mapAsync(GPUMapMode.READ).then(() => {
        const mappedData = new BigUint64Array(syncData.buffer.getMappedRange());
        const [startTimeU64, endTimeU64] = mappedData;
        const [kernelType, kernelName] = kernelInfo;

        syncData.buffer.unmap();

        if (typeof this.backend.queryTimeBase === 'undefined') {
          this.backend.queryTimeBase = startTimeU64;
        }

        const startTime = Number(startTimeU64 - this.backend.queryTimeBase);
        const endTime = Number(endTimeU64 - this.backend.queryTimeBase);

        if (!Number.isSafeInteger(startTime) || !Number.isSafeInteger(endTime)) {
          throw new RangeError('incorrect timestamp range');
        }

        this.backend.gpuDataManager.release(syncData.id);
        if (this.backend.env.webgpu.profiling?.ondata) {
          this.backend.env.webgpu.profiling.ondata({
            version: 1,
            inputsMetadata: inputTensorViews.map(
                value => ({dims: value.dims, dataType: tensorDataTypeEnumToString(value.dataType)})),
            outputsMetadata: outputTensorViews.map(
                value => ({dims: value.dims, dataType: tensorDataTypeEnumToString(value.dataType)})),
            kernelId,
            kernelType,
            kernelName,
            startTime,
            endTime,
          });
        } else {
          // if no callback is provided, print the profiling message to console
          let inputShapes = '';
          inputTensorViews.forEach((value, i) => {
            inputShapes += `input[${i}]: [${value.dims}] | ${tensorDataTypeEnumToString(value.dataType)}, `;
          });
          let outputShapes = '';
          inputTensorViews.forEach((value, i) => {
            outputShapes += `output[${i}]: [${value.dims}] | ${tensorDataTypeEnumToString(value.dataType)}, `;
          });
          // eslint-disable-next-line no-console
          console.log(`[profiling] kernel "${kernelId}|[${kernelType}] ${kernelName}" ${inputShapes}${
              outputShapes}execution time: ${endTime - startTime} ns`);
        }
      });
    }

    if (this.backend.pendingDispatchNumber >= 16) {
      this.backend.flush();
    }
  }
  dispose(): void {
    // this.repo.forEach(a => this.glContext.deleteProgram(a.program));
  }
  build(programInfo: ProgramInfo, normalizedDispatchGroupSize: [number, number, number]): Artifact {
    const device = this.backend.device;
    const extensions: string[] = [];
    if (device.features.has('shader-f16')) {
      extensions.push('enable f16;');
    }
    const shaderHelper = createShaderHelper(normalizedDispatchGroupSize);
    const userCode = programInfo.getShaderSource(shaderHelper);
    const code = `${extensions.join('\n')}\n${shaderHelper.additionalImplementations}\n${userCode}`;
    const shaderModule = device.createShaderModule({code, label: programInfo.name});
    LOG_DEBUG('verbose', () => `[WebGPU] ${programInfo.name} shader code: ${code}`);

    const computePipeline = device.createComputePipeline(
        {compute: {module: shaderModule, entryPoint: 'main'}, layout: 'auto', label: programInfo.name});

    return {programInfo, computePipeline};
  }

  normalizeDispatchGroupSize(dispatchGroup: ReturnType<ProgramInfo['getRunData']>['dispatchGroup']):
      [number, number, number] {
    const x = typeof dispatchGroup === 'number' ? dispatchGroup : dispatchGroup.x;
    const y = typeof dispatchGroup === 'number' ? 1 : (dispatchGroup.y || 1);
    const z = typeof dispatchGroup === 'number' ? 1 : (dispatchGroup.z || 1);
    const limitPerDimension = this.backend.device.limits.maxComputeWorkgroupsPerDimension;
    if (x <= limitPerDimension && y <= limitPerDimension && z <= limitPerDimension) {
      return [x, y, z];
    }
    const size = x * y * z;
    let dispatchAverage = Math.ceil(Math.sqrt(size));
    if (dispatchAverage > limitPerDimension) {
      dispatchAverage = Math.ceil(Math.cbrt(size));
      if (dispatchAverage > limitPerDimension) {
        throw new Error('Total dispatch size exceeds WebGPU maximum.');
      }
      return [dispatchAverage, dispatchAverage, dispatchAverage];
    } else {
      return [dispatchAverage, dispatchAverage, 1];
    }
  }
}
