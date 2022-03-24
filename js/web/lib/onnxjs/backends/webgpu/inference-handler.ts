// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {InferenceHandler} from '../../backend';
import {Tensor} from '../../tensor';

import {WebGpuSessionHandler} from './session-handler';
import {createTensorDataManager, TensorDataManager} from './tensor-data-manager';
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
  // per inference context
  dataManager: TensorDataManager;

  constructor(public session: WebGpuSessionHandler) {
    this.dataManager = createTensorDataManager(session.backend.device);
  }

  private async uploadGpuData(tensor: Tensor, textureType: GpuDataType): Promise<GpuData> {
    if (this.session.isInitializer(tensor.dataId)) {
      return this.session.dataManager.uploadTensorToGpu(tensor, textureType);
    }

    return this.dataManager.uploadTensorToGpu(tensor, textureType);
  }

  private createGpuData(type: Tensor.DataType, dims: readonly number[], gpuDataType: GpuDataType): [Tensor, GpuData] {
    return this.dataManager.createGpuTensor(type, dims, gpuDataType);
  }

  async run(program: ProgramInfoLoader|ProgramInfo, inputs: readonly Tensor[]): Promise<Tensor[]> {
    if (inputs.length !== program.inputTypes.length) {
      throw new Error(`Input size must be equal to ${program.inputTypes.length}.`);
    }

    // create info for input
    const inputDatas: GpuData[] = [];
    for (let i = 0; i < program.inputTypes.length; ++i) {
      inputDatas[i] = await this.uploadGpuData(inputs[i], program.inputTypes[i]);
    }

    const key = getProgramInfoUniqueKey(program, inputDatas);
    let artifact = this.session.programManager.getArtifact(key);
    const programInfo = artifact ?
        artifact.programInfo :
        (typeof (program as ProgramInfoLoader).get === 'function' ? (program as ProgramInfoLoader).get() :
                                                                    (program as ProgramInfo));

    // create texture info for outputs
    const outputDatas: GpuData[] = [];
    const outputTensors: Tensor[] = [];
    for (let i = 0; i < programInfo.outputs.length; ++i) {
      const [tensor, gpuData] = this.createGpuData(
          programInfo.outputs[i].type, programInfo.outputs[i].dims, programInfo.outputs[i].gpuDataType);
      outputTensors.push(tensor);
      outputDatas.push(gpuData);
    }

    if (!artifact) {
      artifact = this.session.programManager.build(programInfo);
      this.session.programManager.setArtifact(key, artifact);
    }

    this.session.programManager.run(artifact, inputDatas, outputDatas, artifact.programInfo.dispatchGroup(inputs));

    return outputTensors;
  }

  reshape(input: Tensor, reshapedDims: readonly number[]): Tensor {
    return this.dataManager.hasGpuData(input.dataId) ?
        this.dataManager.createGpuRef(input.dataId, input.type, reshapedDims)[0] :
        new Tensor(reshapedDims, input.type, undefined, undefined, input.data);
  }

  dispose(): void {}
}
