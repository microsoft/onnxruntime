// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {env} from 'onnxruntime-common';

import {WebGpuBackend} from '../backend-webgpu';

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
  run(buildArtifact: Artifact, inputs: GpuData[], outputs: GpuData[],
      dispatchGroup: {x: number; y?: number; z?: number}): void {
    const device = this.backend.device;
    const computePassEncoder = this.backend.getComputePassEncoder();

    if (this.backend.profilingEnabled) {
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
    const bindGroup = device.createBindGroup({layout: buildArtifact.computePipeline.getBindGroupLayout(0), entries});
    computePassEncoder.setBindGroup(0, bindGroup);

    const {x, y, z} = dispatchGroup;
    computePassEncoder.dispatchWorkgroups(x, y, z);

    this.backend.pendingDispatchNumber++;

    if (this.backend.profilingEnabled) {
      // profiling write end timestamp

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (computePassEncoder as any).writeTimestamp(this.backend.profilingQuerySet, 1);
      // eslint-disable-next-line no-bitwise
      const queryData = this.backend.gpuDataManager.create(16, GPUBufferUsage.COPY_SRC | GPUBufferUsage.QUERY_RESOLVE);
      // eslint-disable-next-line no-bitwise
      const syncData = this.backend.gpuDataManager.create(16, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);

      this.backend.endComputePass();
      this.backend.getCommandEncoder().resolveQuerySet(this.backend.profilingQuerySet, 0, 2, queryData.buffer, 0);
      this.backend.getCommandEncoder().copyBufferToBuffer(queryData.buffer, 0, syncData.buffer, 0, 16);
      this.backend.flush();

      const kernelId = this.backend.currentKernelId!;
      const kernelName = this.backend.kernels.get(kernelId)![0];

      syncData.buffer.mapAsync(GPUMapMode.READ).then(() => {
        const mappedData = new BigUint64Array(syncData.buffer.getMappedRange());
        const startTimeU64 = mappedData[0];
        const endTimeU64 = mappedData[1];

        syncData.buffer.unmap();

        if (typeof this.backend.profilingTimeBase === 'undefined') {
          this.backend.profilingTimeBase = startTimeU64;
        }

        const startTime = Number(startTimeU64 - this.backend.profilingTimeBase);
        const endTime = Number(endTimeU64 - this.backend.profilingTimeBase);

        if (!Number.isSafeInteger(startTime) || !Number.isSafeInteger(endTime)) {
          throw new RangeError('incorrect timestamp range');
        }

        this.backend.gpuDataManager.release(queryData.id);
        this.backend.gpuDataManager.release(syncData.id);

        // eslint-disable-next-line no-console
        console.log(`[profiling] kernel "${kernelId}|${kernelName}" execution time: ${endTime - startTime} ns`);
      });
    }

    if (this.backend.pendingDispatchNumber >= 16) {
      this.backend.flush();
    }
  }
  dispose(): void {
    // this.repo.forEach(a => this.glContext.deleteProgram(a.program));
  }
  build(programInfo: ProgramInfo): Artifact {
    const device = this.backend.device;

    const shaderModule = device.createShaderModule({code: programInfo.shaderSource});
    if (env.debug) {
      // eslint-disable-next-line no-console
      console.log('WebGpuProgram: ' + programInfo.shaderSource);
    }

    const computePipeline =
        device.createComputePipeline({compute: {module: shaderModule, entryPoint: 'main'}, layout: 'auto'});

    return {programInfo, computePipeline};
  }
}
