// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {WebGpuBackend} from '../backend-webgpu';
import {LOG_DEBUG} from '../log';
import {TensorView} from '../tensor-view';

import {createShaderHelper} from './ops/common';
import {Artifact, GpuData, PendingKernelInfo, ProgramInfo, QueryType} from './types';

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
    this.backend.writeTimeStamp(this.backend.pendingDispatchNumber * 2);
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

    if (this.backend.queryType !== QueryType.none) {
      const kernelId = this.backend.currentKernelId!;
      const kernelInfo = this.backend.kernels.get(kernelId)!;
      let kernelName = kernelInfo[0];
      if (buildArtifact.programInfo.name !== kernelName) {
        kernelName = `${kernelName}/${buildArtifact.programInfo.name}`;
      }
      const pendingKernelInfo: PendingKernelInfo = {
        id: kernelId,
        name: kernelName,
        inputTensorViews,
        outputTensorViews,
      };
      this.backend.pendingKernels.push(pendingKernelInfo);
      this.backend.writeTimeStamp(this.backend.pendingDispatchNumber * 2 + 1);
    }

    this.backend.pendingDispatchNumber++;
    if (this.backend.pendingDispatchNumber >= this.backend.maxDispatchNumber ||
        this.backend.queryType === QueryType.atPasses) {
      this.backend.endComputePass();
    }
    if (this.backend.pendingDispatchNumber >= this.backend.maxDispatchNumber) {
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
