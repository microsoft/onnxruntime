// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {Attribute} from '../../../attribute';
import {DepthToSpace} from '../../../ops/depth-to-space';
import {Tensor} from '../../../tensor';
import {WebGLInferenceHandler} from '../inference-handler';
import {Artifact, ProgramInfo, RunData} from '../types';

import {reshape} from './reshape';
import {WebGLTranspose} from './transpose';

export class WebGLDepthToSpace extends DepthToSpace {
  protected transposeProgramInfo: ProgramInfo;

  protected transposeArtifact: Artifact;

  run(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): Tensor[] {
    const programManager = inferenceHandler.session.programManager;
    const transposePerm = this.mode === 'DCR' ? [0, 3, 4, 1, 5, 2] : [0, 1, 4, 2, 5, 3];
    const firstReshapeShape = this.mode === 'DCR' ?
        [
          inputs[0].dims[0], this.blocksize, this.blocksize, inputs[0].dims[1] / this.blocksizeSqr, inputs[0].dims[2],
          inputs[0].dims[3]
        ] :
        [
          inputs[0].dims[0], inputs[0].dims[1] / this.blocksizeSqr, this.blocksize, this.blocksize, inputs[0].dims[2],
          inputs[0].dims[3]
        ];

    const transpose = new WebGLTranspose();
    const attributes = new Attribute(undefined);
    attributes.set('perm', 'ints', transposePerm);
    transpose.initialize(attributes);

    // First reshape

    const firstReshapedTensor = reshape(inferenceHandler, inputs[0], firstReshapeShape);

    // transpose
    if (!this.transposeProgramInfo) {
      this.transposeProgramInfo = transpose.createProgramInfo(inferenceHandler, [firstReshapedTensor]);
      this.transposeArtifact = programManager.build(this.transposeProgramInfo);
    }
    const runDataTranspose =
        transpose.createRunData(inferenceHandler, this.transposeProgramInfo, [firstReshapedTensor]);
    inferenceHandler.checkAndUpdateTextureForm(this.transposeArtifact, runDataTranspose);
    programManager.run(this.transposeArtifact, runDataTranspose);
    const transposeOutput = runDataTranspose.outputTextureData.tensor;

    // Second reshape
    const result = reshape(inferenceHandler, transposeOutput, [
      inputs[0].dims[0], inputs[0].dims[1] / this.blocksizeSqr, inputs[0].dims[2] * this.blocksize,
      inputs[0].dims[3] * this.blocksize
    ]);
    return [result];
  }

  protected getOutShape(input: Tensor): number[] {
    const batchSize = input.dims[0];
    const inputDepth = input.dims[1];
    const inputHeight = input.dims[2];
    const inputWidth = input.dims[3];
    if (inputDepth % (this.blocksizeSqr) !== 0) {
      throw new Error('Input depth must be divisible by squared blocksize.');
    }
    const outputDepth = inputDepth / this.blocksizeSqr;
    const outputHeight = inputHeight * this.blocksize;
    const outputWidth = inputWidth * this.blocksize;
    return [batchSize, outputDepth, outputHeight, outputWidth];
  }

  createRunData(handler: WebGLInferenceHandler, programInfo: ProgramInfo, inputs: Tensor[]): RunData {
    const inputTDs = inputs.map((t, i) => handler.getOrCreateTextureData(t, programInfo.inputLayouts[i]));
    return {
      inputTextureDatas: inputTDs,
      outputTextureData: handler.createTextureDataFromLayout(programInfo.outputLayout, inputTDs[0].tensor.type),
      uniformData: {}
    };
  }
}