// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {InferenceHandler} from '../../backend';
import {createView, Tensor} from '../../tensor';

import {createGpuDataManager, GpuDataManager} from './gpu-data-manager';
import {WebGpuSessionHandler} from './session-handler';
import {GpuData, GpuDataType, ProgramInfo, ProgramInfoLoader} from './types';

const getProgramInfoUniqueKey = (programInfo: ProgramInfo|ProgramInfoLoader, inputGpuDatas: GpuData[]): string => {
  const inputs = inputGpuDatas.map(data => `${data.id}`).join('_');
  let key = programInfo.name;
  if (programInfo.cacheHint) {
    key += '[' + programInfo.cacheHint + ']';
  }
  key += ':' + inputs;
  return key;
};

export class WebGpuInferenceHandler implements InferenceHandler {
  dataManager: GpuDataManager;
  constructor(public session: WebGpuSessionHandler) {
    this.dataManager = createGpuDataManager(session.backend.device);
  }

  private uploadGpuData(tensor: Tensor, textureType: GpuDataType): GpuData {
    if (this.session.isInitializer(tensor.dataId)) {
      return this.session.dataManager.uploadData(tensor, textureType);
    }

    return this.dataManager.uploadData(tensor, textureType);
  }

  private createGpuData(type: Tensor.DataType, dims: readonly number[], gpuDataType: GpuDataType): GpuData {
    return this.dataManager.createData(type, dims, gpuDataType);
  }

  run(program: ProgramInfoLoader|ProgramInfo, inputs: readonly Tensor[]): Tensor[] {
    if (inputs.length !== program.inputTypes.length) {
      throw new Error(`Input size must be equal to ${program.inputTypes.length}.`);
    }

    // create info for input
    const inputDatas: GpuData[] = [];
    for (let i = 0; i < program.inputTypes.length; ++i) {
      inputDatas[i] = this.uploadGpuData(inputs[i], program.inputTypes[i]);
    }

    const key = getProgramInfoUniqueKey(program, inputDatas);
    let artifact = this.session.programManager.getArtifact(key);
    const programInfo = artifact ?
        artifact.programInfo :
        (typeof (program as ProgramInfoLoader).get === 'function' ? (program as ProgramInfoLoader).get() :
                                                                    (program as ProgramInfo));

    // create texture info for outputs
    const outputDatas: GpuData[] = [];
    for (let i = 0; i < programInfo.outputs.length; ++i) {
      outputDatas.push(this.createGpuData(
          programInfo.outputs[i].type, programInfo.outputs[i].dims, programInfo.outputs[i].gpuDataType));
    }

    if (!artifact) {
      artifact = this.session.programManager.build(programInfo);
      this.session.programManager.setArtifact(key, artifact);
    }

    this.session.programManager.run(artifact, inputDatas, outputDatas, artifact.programInfo.dispatchGroup(inputs));

    const outputTensors: Tensor[] = [];
    for (let i = 0; i < outputDatas.length; i++) {
      const outputTensorInfo = artifact.programInfo.outputs[i];
      const dims = outputTensorInfo.dims;
      const type = outputTensorInfo.type;
      const outputData = outputDatas[i];
      const tensor = new Tensor(dims, type, undefined, async () => {
        const data = await this.dataManager.downloadData(outputData.id);
        return createView(data, type);
      }, undefined, outputData.id);
      outputTensors.push(tensor);
    }
    return outputTensors;
  }

  dispose(): void {}
}
