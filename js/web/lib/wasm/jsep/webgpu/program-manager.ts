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
    computePassEncoder.dispatch(x, y, z);

    this.backend.pendingDispatchNumber++;

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

    const computePipeline = device.createComputePipeline({compute: {module: shaderModule, entryPoint: 'main'}});

    return {programInfo, computePipeline};
  }
}
