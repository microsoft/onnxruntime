// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {InstanceNormalization} from '../../../ops/instance-normalization';
import {Tensor} from '../../../tensor';
import {getGlsl} from '../glsl-source';
import {WebGLInferenceHandler} from '../inference-handler';
import {Artifact, ProgramInfo, RunData, TextureLayout} from '../types';

export class WebGLInstanceNormalization extends InstanceNormalization {
  run(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): Tensor[] {
    if (!this.artifacts) {
      this.artifacts = [];
      const programInfos = this.createProgramInfos(inferenceHandler, inputs);
      programInfos.forEach((pi) => {
        const artifact = inferenceHandler.session.programManager.build(pi);
        this.artifacts.push(artifact);
      });
    }

    const runDatas = this.createRunDatas(inferenceHandler, this.artifacts.map(a => a.programInfo), inputs);
    runDatas.forEach((v, i) => inferenceHandler.session.programManager.run(this.artifacts[i], v));
    return [runDatas[1].outputTextureData.tensor];
  }

  checkInputTypes(inputs: Tensor[]): boolean {
    if (!super.checkInputTypes(inputs)) {
      return false;
    }

    if (inputs[0].dims.length !== 4) {
      // currently webgl implementation only support 4-D input.
      return false;
    }

    return true;
  }

  createMeanAndVarianceProgramInfo(inferenceHandler: WebGLInferenceHandler, xLayout: TextureLayout): ProgramInfo {
    const xDims = xLayout.shape;
    const channel = xDims[1];
    const channelSize = xDims[2] * xDims[3];
    const outputShape = [xDims[0], channel];
    const outputUnpackedShape = [xDims[0], channel * 4];

    const shaderSource = `
    vec4 process(int[2] indices) {
      vec4 v = vec4(0.0);
      int a[4];
      a[0] = indices[0];
      a[1] = indices[1];
      float temp = 0.0;
      for(int a2=0; a2<${xDims[2]}; a2++) {
        a[2] = a2;
        for(int a3=0; a3<${xDims[3]}; a3++) {
          a[3] = a3;
          float x = _X(a);
          temp += x;
        }
      }
      float mean = temp / float(${channelSize});
      temp = 0.0;
      for(int a2=0; a2<${xDims[2]}; a2++) {
        a[2] = a2;
        for(int a3=0; a3<${xDims[3]}; a3++) {
          a[3] = a3;
          float x = _X(a);
          temp += (x - mean) * (x - mean);
        }
      }
      v.r = mean;
      v.g = temp / float(${channelSize});

      return v;
    }`;
    return {
      inputLayouts: [xLayout],
      outputLayout: inferenceHandler.createTextureLayoutFromShape(outputShape, 4, outputUnpackedShape),
      samplers: ['X'],
      shaderSource,
    };
  }

  createComputOutputProgramInfo(
      inferenceHandler: WebGLInferenceHandler, xLayout: TextureLayout, scaleLayout: TextureLayout,
      bLayout: TextureLayout, meanAndVarianceLayout: TextureLayout): ProgramInfo {
    const glsl = getGlsl(inferenceHandler.session.backend.glContext.version);
    const shaderSource = `
    vec4 get_MeanAndVariance(int[2] mv) {
      int offset = indicesToOffset_MeanAndVariance(mv);
      vec2 coords = offsetToCoords(offset, ${meanAndVarianceLayout.width}, ${meanAndVarianceLayout.height});
      return ${glsl.texture2D}(MeanAndVariance, coords);
    }

    float process(int[4] indices) {

          int mv[2];
          mv[0] = indices[0];
          mv[1] = indices[1];
          vec4 mean_and_variance = get_MeanAndVariance(mv);
          float mean = mean_and_variance.r;
          float variance = mean_and_variance.g;

          int sb[1];
          sb[0] = indices[1];
          float scale = _Scale(sb);
          float b = _B(sb);

          return scale * (_X(indices) - mean) / sqrt(variance + epsilon) + b;
        }`;
    return {
      inputLayouts: [xLayout, meanAndVarianceLayout, scaleLayout, bLayout],
      outputLayout: inferenceHandler.createTextureLayoutFromShape(xLayout.shape),
      samplers: ['X', 'MeanAndVariance', 'Scale', 'B'],
      variables: [{name: 'epsilon', type: 'float'}],
      shaderSource,
    };
  }
  createProgramInfos(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): ProgramInfo[] {
    const xLayout = inferenceHandler.getOrCreateTextureLayout(inputs[0]);
    const scaleLayout = inferenceHandler.getOrCreateTextureLayout(inputs[1]);
    const bLayout = inferenceHandler.getOrCreateTextureLayout(inputs[2]);
    const meanAndVarianceProgramInfo = this.createMeanAndVarianceProgramInfo(inferenceHandler, xLayout);
    const computeOutputProgramInfo = this.createComputOutputProgramInfo(
        inferenceHandler, xLayout, scaleLayout, bLayout, meanAndVarianceProgramInfo.outputLayout);

    const programInfos: ProgramInfo[] = [meanAndVarianceProgramInfo, computeOutputProgramInfo];
    return programInfos;
  }
  createRunDatas(inferenceHandler: WebGLInferenceHandler, programInfos: ProgramInfo[], inputs: Tensor[]): RunData[] {
    const dataType = inputs[0].type;
    const inputTD = inferenceHandler.getOrCreateTextureData(inputs[0], programInfos[0].inputLayouts[0]);
    const scaleTD = inferenceHandler.getOrCreateTextureData(inputs[1], programInfos[1].inputLayouts[2]);
    const bTD = inferenceHandler.getOrCreateTextureData(inputs[2], programInfos[1].inputLayouts[3]);
    const runDatas: RunData[] = [];
    runDatas.push({
      inputTextureDatas: [inputTD],
      outputTextureData: inferenceHandler.createTextureDataFromLayout(programInfos[0].outputLayout, dataType),
      uniformData: {}
    });
    runDatas.push({
      inputTextureDatas: [inputTD, runDatas[0].outputTextureData, scaleTD, bTD],
      outputTextureData: inferenceHandler.createTextureDataFromLayout(programInfos[1].outputLayout, dataType),
      uniformData: {'epsilon': this.epsilon}
    });
    return runDatas;
  }
  protected artifacts: Artifact[];
}
