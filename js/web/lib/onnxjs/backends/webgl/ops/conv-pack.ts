// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {Attribute} from '../../../attribute';
import {Logger} from '../../../instrument';
import {Conv} from '../../../ops/conv';
import {Tensor} from '../../../tensor';
import {assert, PoolConvUtil} from '../../../util';
import {WebGLInferenceHandler} from '../inference-handler';
import {Artifact, ProgramInfo} from '../types';
import {WebGLConv} from './conv';
import {WebGLIm2ColPacked} from './im2col-pack';
import {WebGLMatMulPacked} from './matmul-pack';
import {WebGLReshapePacked} from './reshape-packed';

export class WebGLConvPacked extends Conv {
  protected artifacts: Artifact[];
  protected programInfo: ProgramInfo[];

  run(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): Tensor[] {
    const programManager = inferenceHandler.session.programManager;
    const xshape = inputs[0].dims.slice();
    const kshape = inputs[1].dims.slice();
    // if kernelShape is not specified in the attributes of this op, infer it from the weight tensor dims
    if (this.kernelShape.length === 0) {
      for (let i = 2; i < kshape.length; ++i) {
        this.kernelShape.push(kshape[i]);
      }
    }
    PoolConvUtil.adjustPadsBasedOnAutoPad(
        inputs[0].dims, this.strides, this.dilations, this.kernelShape, this.pads, this.autoPad);
    Logger.verbose(
        'Conv',
        `autpPad:${this.autoPad}, dilations:${this.dilations}, group:${this.group}, kernelShape:${
            this.kernelShape}, pads:${this.pads}, strides:${this.strides}`);

    const outputShape = WebGLConv.calcOutputShape(xshape, kshape, this.dilations, this.pads, this.strides);
    const im2col = new WebGLIm2ColPacked(outputShape, kshape, this.dilations, this.pads, this.strides);
    const matmul = new WebGLMatMulPacked();
    if (!!this.activation) {
      const attributes = new Attribute(undefined);
      attributes.set('__internal_activation', 'string', (this.activation));
      matmul.initialize(attributes);
    }
    const reshape = new WebGLReshapePacked();
    // shape for kernel reshape
    const shape =
        new Tensor([2], 'int32', undefined, undefined, new Int32Array([kshape[0], kshape[1] * kshape[2] * kshape[3]]));
    if (!this.artifacts) {
      this.artifacts = [];
      this.programInfo = [];
      this.programInfo[0] = im2col.createProgramInfo(inferenceHandler, [inputs[0], inputs[1]]);
      this.artifacts[0] = programManager.build(this.programInfo[0]);

      this.programInfo[1] = reshape.createProgramInfo(inferenceHandler, [inputs[1], shape]);
      this.artifacts[1] = programManager.build(this.programInfo[1]);
    }

    // run im2col
    const runDataIm2col = im2col.createRunData(inferenceHandler, this.programInfo[0], [inputs[0], inputs[1]]);
    inferenceHandler.checkAndUpdateTextureForm(this.artifacts[0], runDataIm2col);
    programManager.run(this.artifacts[0], runDataIm2col);
    const im2colOutput = runDataIm2col.outputTextureData.tensor;

    // reshape kernel
    const runDataKernelReshape = reshape.createRunData(inferenceHandler, this.programInfo[1], [inputs[1], shape]);
    inferenceHandler.checkAndUpdateTextureForm(this.artifacts[1], runDataKernelReshape);
    programManager.run(this.artifacts[1], runDataKernelReshape);
    const kernelReshaped = runDataKernelReshape.outputTextureData.tensor;

    // run matmul
    const hasBias = (inputs.length === 3);
    assert(this.artifacts.length > 1, () => 'expect at least 2 artifacts created');
    if (this.artifacts.length === 2) {
      this.programInfo[2] = matmul.createProgramInfo(
          inferenceHandler, hasBias ? [kernelReshaped, im2colOutput, inputs[2]] : [kernelReshaped, im2colOutput]);
      this.artifacts[2] = programManager.build(this.programInfo[2]);
    }
    const runDataMatmul = matmul.createRunData(
        inferenceHandler, this.programInfo[2],
        hasBias ? [kernelReshaped, im2colOutput, inputs[2]] : [kernelReshaped, im2colOutput]);
    inferenceHandler.checkAndUpdateTextureForm(this.artifacts[2], runDataMatmul);
    programManager.run(this.artifacts[2], runDataMatmul);
    const matmulOutput = runDataMatmul.outputTextureData.tensor;

    // reshape output
    const outputShapeTensor = new Tensor(
        [outputShape.length], 'int32', undefined, undefined,
        new Int32Array([outputShape[0], outputShape[1], outputShape[2], outputShape[3]]));

    assert(this.artifacts.length > 2, () => 'expect at least 3 artifacts created');
    if (this.artifacts.length === 3) {
      this.programInfo[3] = reshape.createProgramInfo(inferenceHandler, [matmulOutput, outputShapeTensor]);
      this.artifacts[3] = programManager.build(this.programInfo[3]);
    }
    const runDataOutputReshape =
        reshape.createRunData(inferenceHandler, this.programInfo[3], [matmulOutput, outputShapeTensor]);
    inferenceHandler.checkAndUpdateTextureForm(this.artifacts[3], runDataOutputReshape);
    programManager.run(this.artifacts[3], runDataOutputReshape);
    return [runDataOutputReshape.outputTextureData.tensor];
  }
}
