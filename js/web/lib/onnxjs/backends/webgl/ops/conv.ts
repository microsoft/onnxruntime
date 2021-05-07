// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {Attribute} from '../../../attribute';
import {Logger} from '../../../instrument';
import {Conv} from '../../../ops/conv';
import {Tensor} from '../../../tensor';
import {PoolConvUtil} from '../../../util';
import {getGlsl} from '../glsl-source';
import {WebGLInferenceHandler} from '../inference-handler';
import {Artifact, ProgramInfo, RunData, TextureLayout, WebGLOperator} from '../types';
import {WebGLContext} from '../webgl-context';

import {WebGLConvPacked} from './conv-pack';
import {getActicationSnippet} from './fuse-utils';

export class WebGLConv extends Conv {
  unpackedGroupedConvImpl: WebGLUnpackedGroupedConv;
  unpackedConvImpl: WebGLUnpackedConv;
  packedConvImpl: WebGLConvPacked;

  constructor() {
    super();
    this.unpackedGroupedConvImpl = new WebGLUnpackedGroupedConv();
    this.unpackedConvImpl = new WebGLUnpackedConv();
    this.packedConvImpl = new WebGLConvPacked();
  }

  initialize(attributes: Attribute): void {
    super.initialize(attributes);
    this.unpackedGroupedConvImpl.initialize(attributes);
    this.unpackedConvImpl.initialize(attributes);
    this.packedConvImpl.initialize(attributes);
  }

  run(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): Tensor[] {
    const packMode = inferenceHandler.session.pack;
    if (this.group > 1) {
      return this.unpackedGroupedConvImpl.run(inferenceHandler, inputs);
    } else if (packMode && inputs[0].dims.length === 4 && inputs[0].dims[0] === 1) {
      return this.packedConvImpl.run(inferenceHandler, inputs);
    } else {
      return this.unpackedConvImpl.run(inferenceHandler, inputs);
    }
  }

  static calcOutputShape(
      inputShape: number[], kernelShape: number[], dilations: number[], adjustPads: number[],
      strides: number[]): number[] {
    const batchSize = inputShape[0];
    const inputSpatialShape = inputShape.slice(2);
    const spatialRank = inputSpatialShape.length;
    const outChannels = kernelShape[0];
    const kernelSpatialShape = kernelShape.slice(2);
    const dilatedKernelShape = kernelSpatialShape.map((v, i) => v + (v - 1) * (dilations[i] - 1));
    const inputSpatialShapeWithPad = inputSpatialShape.map((v, i) => v + adjustPads[i] + adjustPads[i + spatialRank]);
    const outputSpatialShape =
        inputSpatialShapeWithPad.map((v, i) => Math.floor((v - dilatedKernelShape[i] + strides[i]) / strides[i]));
    const outputShape = [batchSize, outChannels].concat(...outputSpatialShape);
    return outputShape;
  }
}

export class WebGLUnpackedGroupedConv extends Conv implements WebGLOperator {
  run(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): Tensor[] {
    return inferenceHandler.run(this, inputs);
  }

  createProgramInfo(handler: WebGLInferenceHandler, inputs: Tensor[]): ProgramInfo {
    const hasBias = inputs.length > 2;
    const processBias = hasBias ? 'value += getBias(output_channel);' : '';
    const xShape = inputs[0].dims.slice();
    const wShape = inputs[1].dims.slice();
    const outputChannelsPerGroup = wShape[0] / this.group;
    // if kernelShape is not specified in the attributes of this op, infer it from the weight tensor dims
    if (this.kernelShape.length === 0) {
      for (let i = 2; i < wShape.length; ++i) {
        this.kernelShape.push(wShape[i]);
      }
    }
    PoolConvUtil.adjustPadsBasedOnAutoPad(
        inputs[0].dims, this.strides, this.dilations, this.kernelShape, this.pads, this.autoPad);
    Logger.verbose(
        'Conv',
        `autpPad:${this.autoPad}, dilations:${this.dilations}, group:${this.group}, kernelShape:${
            this.kernelShape}, pads:${this.pads}, strides:${this.strides}`);
    const outputShape = WebGLConv.calcOutputShape(xShape, wShape, this.dilations, this.pads, this.strides);
    const glsl = getGlsl(handler.session.backend.glContext.version);

    const {activationFunction, applyActivation} = getActicationSnippet(this.activation);

    const shaderSource = `
    const ivec2 strides = ivec2(${this.strides[0]}, ${this.strides[1]});
    const ivec2 pads = ivec2(${this.pads[0]}, ${this.pads[1]});
    ${activationFunction}
    void main() {
      ivec4 coords = getOutputCoords();
      int batch = coords.x;
      int output_channel = coords.y;
      ivec2 xRCCorner = coords.zw * strides - pads;
      int group_id = output_channel / ${outputChannelsPerGroup};

      float value = 0.0;
      for (int wInChannel = 0; wInChannel < ${wShape[1]}; wInChannel++) {
        int input_channel = group_id * ${wShape[1]} + wInChannel;
        for (int wHeight = 0; wHeight < ${wShape[2]}; wHeight++) {
          int xHeight = xRCCorner.x + wHeight * ${this.dilations[0]};

          if (xHeight < 0 || xHeight >= ${xShape[2]}) {
            continue;
          }

          for (int wWidth = 0; wWidth < ${wShape[3]}; wWidth++) {
            int xWidth = xRCCorner.y + wWidth * ${this.dilations[1]};
            if (xWidth < 0 || xWidth >= ${xShape[3]}) {
              continue;
            }

            float xVal = getX(batch, input_channel, xWidth, xHeight);
            float wVal = getW(output_channel, wInChannel, wWidth, wHeight);
            value += xVal*wVal;
          }
        }
      }
      ${processBias}
      ${applyActivation}
      ${glsl.output} = vec4(value, .0, .0, .0);
    }
`;
    return {
      inputLayouts: inputs.map(t => handler.getOrCreateTextureLayout(t)),
      outputLayout: handler.createTextureLayoutFromShape(outputShape),
      samplers: hasBias ? ['X', 'W', 'Bias'] : ['X', 'W'],
      shaderSource,
      hasMain: true,
    };
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

export class WebGLUnpackedConv extends Conv {
  run(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): Tensor[] {
    const programManager = inferenceHandler.session.programManager;
    if (!this.artifacts) {
      this.artifacts = [];
      const programInfos = this.createProgramInfoArray(inferenceHandler, inputs);
      for (let i = 0; i < programInfos.length; ++i) {
        const artifact = inferenceHandler.session.programManager.build(programInfos[i]);
        this.artifacts.push(artifact);
      }
    }
    const runDataArray = this.createRunDataArray(inferenceHandler, this.artifacts.map(a => a.programInfo), inputs);
    inferenceHandler.checkAndUpdateTextureForm(this.artifacts[0], runDataArray[0]);
    programManager.run(this.artifacts[0], runDataArray[0]);
    programManager.run(this.artifacts[1], runDataArray[1]);
    return [runDataArray[1].outputTextureData.tensor];
  }
  createProgramInfoArray(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): ProgramInfo[] {
    const xshape = inputs[0].dims.slice();
    const kshape = inputs[1].dims.slice();
    // if kernelShape is not specified in the attributes of this op, infer it from the weight tensor dims
    if (this.kernelShape.length === 0) {
      const wDims = inputs[1].dims;
      for (let i = 2; i < wDims.length; ++i) {
        this.kernelShape.push(wDims[i]);
      }
    }
    PoolConvUtil.adjustPadsBasedOnAutoPad(
        inputs[0].dims, this.strides, this.dilations, this.kernelShape, this.pads, this.autoPad);
    Logger.verbose(
        'Conv',
        `autpPad:${this.autoPad}, dilations:${this.dilations}, group:${this.group}, kernelShape:${
            this.kernelShape}, pads:${this.pads}, strides:${this.strides}`);
    const outputShape = WebGLConv.calcOutputShape(xshape, kshape, this.dilations, this.pads, this.strides);
    const im2colProgramInfo = this.createIm2ColProgramInfo(inferenceHandler, inputs, outputShape);
    const dotProductProgramInfo =
        this.createDotProductProgramInfo(inferenceHandler, im2colProgramInfo.outputLayout, inputs, outputShape);
    return [im2colProgramInfo, dotProductProgramInfo];
  }
  createRunDataArray(inferenceHandler: WebGLInferenceHandler, programInfos: ProgramInfo[], inputs: Tensor[]):
      RunData[] {
    const k = inputs[1];
    const b = inputs.length >= 3 ? inputs[2] : undefined;
    let kTD = inferenceHandler.getTextureData(k.dataId);
    if (!kTD) {
      Logger.verbose('Conv', 'Did not find the adjustedKernel texture in the cache. Creating rew.');
      const newKernelData =
          WebGLUnpackedConv.prepKernelForDotProduct(k.dims.slice(), this.group, 4, k.floatData as Float32Array);
      // hack: should use graph transformer to rewrite initializer K
      kTD = inferenceHandler.createTextureDataFromLayoutBindTensor(
          programInfos[1].inputLayouts[1], k.type, newKernelData, k);
    }
    const runtDataIm2Col = {
      inputTextureDatas: [inferenceHandler.getOrCreateTextureData(inputs[0])],
      outputTextureData: inferenceHandler.createTextureDataFromLayout(programInfos[0].outputLayout, inputs[0].type),
      uniformData: {}
    };
    const inputTDs = [runtDataIm2Col.outputTextureData, kTD];
    if (b) {
      inputTDs.push(inferenceHandler.getOrCreateTextureData(b));
    }
    const outputTD = inferenceHandler.createTextureDataFromLayout(programInfos[1].outputLayout, inputs[0].type);
    const runDataDotProduct = {
      inputTextureDatas: inputTDs,
      outputTextureData: outputTD,
      uniformData: {},
      draw: (glContext: WebGLContext, artifact: Artifact) => {
        const gl = glContext.gl;
        const sharedDim = artifact.programInfo.params!.sharedDim as number;
        const sharedDimReadSize = artifact.programInfo.params!.sharedDimReadSize as number;
        const sharedDimOffsetLocation = artifact.uniformLocations.find(l => l.name === 'sharedDimOffset')!.location;
        let blend = false;
        for (let k = 0; k < sharedDim; k += sharedDimReadSize) {
          Logger.verbose('MatMul2D', `k = ${k}, sharedDim: ${sharedDim}, readSize = ${sharedDimReadSize}`);
          if (k === sharedDimReadSize) {
            blend = true;
            gl.enable(gl.BLEND);
            glContext.checkError();
            gl.blendEquation(gl.FUNC_ADD);
            glContext.checkError();
            gl.blendFunc(gl.ONE, gl.ONE);
            glContext.checkError();
          }

          gl.uniform1i(sharedDimOffsetLocation, k);
          glContext.checkError();
          glContext.draw();
        }

        if (blend) {
          gl.disable(gl.BLEND);
          glContext.checkError();
        }
      }
    };
    return [runtDataIm2Col, runDataDotProduct];
  }
  createIm2ColProgramInfo(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[], outputShape: number[]):
      ProgramInfo {
    const xshape = inputs[0].dims.slice();
    const kshape = inputs[1].dims.slice();

    const rank = outputShape.length;
    const im2colDims = WebGLUnpackedConv.calcIm2ColDims(xshape, kshape, outputShape, 4);
    const outputLayout = inferenceHandler.createTextureLayoutFromShape(
        im2colDims, 4, [im2colDims[0], im2colDims[1], im2colDims[2], im2colDims[3] * 4], {breakAxis: 3});

    const shaderSource = `
      const int XC = ${xshape[1]};
      const int XH = ${xshape[2]};
      const int XW = ${xshape[3]};
      const int KH = ${this.kernelShape[0]};
      const int KW = ${this.kernelShape[1]};
      const int dilationH = ${this.dilations[0]};
      const int dilationW = ${this.dilations[1]};
      const int strideH = ${this.strides[0]};
      const int strideW = ${this.strides[1]};
      const int padH = ${this.pads[0]};
      const int padW = ${this.pads[1]};
      const int KHKW = KH*KW;
      const int XCKHKW = XC * KHKW;
      const int outputChannels = 4;
      vec4 process(int indices[${rank}]) {
        int b  = indices[0]; // batch size
        int oh = indices[1] * strideH - padH; //output height
        int ow = indices[2] * strideW - padW; //output width
        int p = indices[3] * outputChannels; //patch
        vec4 value = vec4(0.0);
        for(int i=0; i < outputChannels; ++i) {
          if(p < XCKHKW) {
            int patchC = p / KHKW;
            int patchH = (p - patchC*KHKW) / KW;
            int patchW = (p - patchC*KHKW) - patchH * KW;
            int xh2 = oh + patchH * dilationH;
            int xw2 = ow + patchW * dilationW;
            int x[${xshape.length}];
            x[0] = b;
            x[1] = patchC;
            x[2] = xh2;
            x[3] = xw2;
            if(xh2 >= 0 &&
                xh2 < XH &&
                xw2 >= 0 &&
                xw2 < XW) {
              value[i] = _X(x);
            }
          }
          ++p;
        }
        return value;
      }
      `;
    return {
      inputLayouts: [inferenceHandler.createTextureLayoutFromShape(xshape)],
      outputLayout,
      samplers: ['X'],
      shaderSource,
    };
  }
  createDotProductProgramInfo(
      inferenceHandler: WebGLInferenceHandler, im2colLayout: TextureLayout, inputs: Tensor[],
      outputShape: number[]): ProgramInfo {
    const xshape = inputs[0].dims.slice();
    const kshape = inputs[1].dims.slice();
    const adjustedKernelShape = [kshape[0], Math.ceil((xshape[1] * kshape[2] * kshape[3]) / 4)];
    const kLayout = inferenceHandler.createTextureLayoutFromShape(
        adjustedKernelShape, 4, [adjustedKernelShape[0], adjustedKernelShape[1] * 4], {breakAxis: 1});

    let bLayout: TextureLayout|undefined;
    const rank = outputShape.length;

    const inputLayouts = [im2colLayout, kLayout];
    if (inputs.length === 3) {
      bLayout = inferenceHandler.createTextureLayoutFromShape(inputs[2].dims.slice());
      inputLayouts.push(bLayout);
    }
    const outputLayout = inferenceHandler.createTextureLayoutFromShape(outputShape);
    const initValue = (inputs.length < 3) ? '0.0' : '_B(b)';
    const sharedDim = im2colLayout.shape[3];
    const blendEnabled = inferenceHandler.session.backend.glContext.isBlendSupported && !this.activation;
    const sharedDimReadSize = blendEnabled && inferenceHandler.session.backend.matmulMaxBatchSize ?
        this.calcSharedDimReadSize(inferenceHandler.session.backend.matmulMaxBatchSize, sharedDim) :
        sharedDim;
    const samplers = ['Im2Col', 'K'];
    if (inputs.length === 3) {
      samplers.push('B');
    }

    const {activationFunction, applyActivation} = getActicationSnippet(this.activation);

    const glsl = getGlsl(inferenceHandler.session.backend.glContext.version);
    const shaderSource = `
    ${activationFunction}
    float process(int indices[${rank}]) {
      int b[1];
      b[0] = indices[1];
      int im2col[${im2colLayout.shape.length}];
      im2col[0] = indices[0];
      im2col[1] = indices[2];
      im2col[2] = indices[3];
      int im2colOffset = im2col[0] * ${im2colLayout.strides[0]} + im2col[1] * ${
        im2colLayout.strides[1]} + im2col[2] * ${im2colLayout.strides[2]} + sharedDimOffset;
      int kernelOffset = indices[1] * ${kLayout.strides[0]} + sharedDimOffset;
      float value = sharedDimOffset == 0 ? ${initValue} : 0.0;
      for (int i = 0; i < ${sharedDimReadSize}; ++i) {
        vec2 im2colCoords = offsetToCoords(im2colOffset, ${im2colLayout.width}, ${im2colLayout.height});
        vec2 kernelCoords = offsetToCoords(kernelOffset, ${kLayout.width}, ${kLayout.height});
        value += dot(${glsl.texture2D}(Im2Col, im2colCoords), ${glsl.texture2D}(K, kernelCoords));
        ++im2colOffset;
        ++kernelOffset;
      }
      ${applyActivation}
      return value;
    }`;
    return {
      inputLayouts: inputs.length === 3 ? [im2colLayout, kLayout, bLayout!] : [im2colLayout, kLayout],
      outputLayout,
      shaderSource,
      samplers,
      variables: [{name: 'sharedDimOffset', type: 'int'}],
      params: {sharedDim, sharedDimReadSize}
    };
  }
  static prepKernelForDotProduct(shape: number[], group: number, channels: number, kernel: Float32Array): Float32Array {
    if (group === 1 && (channels === 1 || (shape[2] * shape[3]) % channels === 0)) {
      return kernel;
    }
    const numFeatureMaps = shape[0];
    const oldRowSize = shape[1] * shape[2] * shape[3];
    const newRowSize = Math.ceil(oldRowSize * group / channels) * channels;
    const newSize = numFeatureMaps * newRowSize;
    const buffer = new Float32Array(newSize);
    for (let f = 0; f < numFeatureMaps; ++f) {
      const oldOffset = f * oldRowSize;
      const newOffset = f * newRowSize + f % group * oldRowSize;
      buffer.set(kernel.subarray(oldOffset, oldOffset + oldRowSize), newOffset);
    }
    return buffer;
  }
  static calcIm2ColDims(inputShape: number[], kernelShape: number[], outputShape: number[], channels = 1): number[] {
    return [
      outputShape[0], outputShape[2], outputShape[3],
      Math.ceil(inputShape[1] * kernelShape[2] * kernelShape[3] / channels)
    ];
  }

  protected calcSharedDimReadSize(preferredBatchSize: number, sharedDim: number): number {
    if (preferredBatchSize <= 0 || sharedDim < preferredBatchSize || sharedDim % preferredBatchSize !== 0) {
      return sharedDim;
    }
    return preferredBatchSize;
  }
  protected calcBlockSize(outputLayout: TextureLayout): [number, number]|undefined {
    const preferredRowCount = 64;
    const preferredColCount = 64;
    if (outputLayout.height < preferredRowCount) {
      return undefined;
    }
    return [preferredColCount, preferredRowCount];
  }
  protected artifacts: Artifact[];
  protected readSize = 8;
  protected blockSize = 64;
}
