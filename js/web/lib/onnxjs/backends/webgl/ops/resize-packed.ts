// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {Upsample} from '../../../ops/upsample';
import {Tensor} from '../../../tensor';
import {getGlsl, Glsl} from '../glsl-source';
import {WebGLInferenceHandler} from '../inference-handler';
import {Artifact, ProgramInfo, RunData, TextureLayout, WebGLOperator} from '../types';
import {getCoordsDataType} from '../utils';

import {unpackFromChannel} from './packing_utils';

export class WebGLResizePacked extends Upsample implements WebGLOperator {
  run(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): Tensor[] {
    return inferenceHandler.run(this, inputs);
  }
  createProgramInfo(handler: WebGLInferenceHandler, inputs: Tensor[]): ProgramInfo {
    const inputLayout = handler.getOrCreateTextureLayout(inputs[0], 4, true, inputs[0].dims, true);

    const [roi, scales, outputShape] = this.prepareInputs(inputs);

    const outputLayout =
        handler.createTextureLayoutFromShape(outputShape, 4, outputShape, {isPacked: true, reverseWH: true});

    const glsl = getGlsl(handler.session.backend.glContext.version);
    return createResizeProgramInfo(
        glsl, this.mode, inputLayout, outputLayout, scales, roi, this.useExtrapolation, this.extrapolationValue,
        this.cubicCoefficientA, this.excludeOutside, this.coordinateTransformMode);
  }

  createRunData(handler: WebGLInferenceHandler, programInfo: ProgramInfo, inputs: Tensor[]): RunData {
    const inputTD =
        handler.getOrCreateTextureData(inputs[0], handler.getOrCreateTextureLayout(inputs[0], 1, false, [], true));
    const outputTD = handler.createTextureDataFromLayout(programInfo.outputLayout, inputTD.tensor.type);
    return {inputTextureDatas: [inputTD], outputTextureData: outputTD, uniformData: {}};
  }

  protected artifacts: Artifact[];
}

function createResizeProgramInfo(
    glsl: Glsl, mode: string, inputLayout: TextureLayout, outputLayout: TextureLayout, scales: readonly number[],
    roi: readonly number[], extrapolationEnabled: boolean, extrapolationValue: number, cubicCoefficientA: number,
    excludeOutside: boolean, coordinateTransformMode: string): ProgramInfo {
  const isSame = scales.every(s => s === 1) && coordinateTransformMode !== 'tf_crop_and_resize';
  if (isSame) {
    return {
      inputLayouts: [inputLayout],
      outputLayout,
      samplers: ['X'],
      hasMain: true,
      shaderSource: `void main() {
      vec4 v = ${glsl.texture2D}(X, TexCoords);
      ${glsl.output} = v;
    }`
    };
  }
  const outputShape = outputLayout.unpackedShape;
  const dim = outputShape.length;
  if (dim < 2) {
    throw new Error(`output dimension should be at least 2, but got ${dim}`);
  }

  const outputHeight = outputShape[dim - 2];
  const outputWidth = outputShape[dim - 1];

  const inputShape = inputLayout.unpackedShape;
  if (dim !== inputShape.length) {
    throw new Error(`output dimension should match input ${inputShape.length}, but got ${dim}`);
  }
  const inputHeight = inputShape[dim - 2];
  const inputWidth = inputShape[dim - 1];

  const scalesHeight = scales[dim - 2];
  const scalesWidth = scales[dim - 1];

  let getSourceFracIndex = '';

  if (mode !== 'linear') {
    // TODO: support other modes
    throw new Error(`resize (packed) does not support mode: '${mode}'`);
  }
  switch (coordinateTransformMode) {
    case 'asymmetric':
      getSourceFracIndex = `
        vec4 getSourceFracIndex(ivec4 coords){
          return vec4(coords) / scaleWHWH;
        }
    `;
      break;
    case 'half_pixel':
      getSourceFracIndex = `
        vec4 getSourceFracIndex(ivec4 coords){
          return (vec4(coords) + 0.5) / scaleWHWH - 0.5;
        }
    `;
      break;
    case 'align_corners':
      getSourceFracIndex = `
        vec4 getSourceFracIndex(ivec4 coords){
          vec4 resized = vec4(${outputWidth}.0 - 1.0, ${outputHeight}.0 - 1.0, ${outputWidth}.0 - 1.0, ${
          outputHeight}.0 - 1.0);
          vec4 original = vec4(${inputWidth}.0 - 1.0, ${inputHeight}.0 - 1.0, ${inputWidth}.0 - 1.0, ${
          inputHeight}.0 - 1.0);
          vec4 new_scale = original / resized;
          return vec4(coords) * new_scale;
        }
      `;
      break;
    default:
      // TODO:supporting other coordinateTransformModes
      throw new Error(`resize (packed) does not support coordinateTransformMode: '${coordinateTransformMode}'`);
  }

  const coordsDataType = getCoordsDataType(dim);
  const unpackChannel = unpackFromChannel();
  const shader = `
        const vec2 inputWH = vec2(${inputHeight}.0, ${inputWidth}.0);
        const vec4 scaleWHWH = vec4(${scalesHeight}.0, ${scalesWidth}.0, ${scalesHeight}.0, ${scalesWidth}.0);
        ${unpackChannel}
        ${getSourceFracIndex}
        float getAValue(int x10, int r, int c, int d) {
          return getChannel(getA(x10, r, c, d), vec2(c, d));
        }
        void main() {
          ${coordsDataType} rc = getOutputCoords();

          int batch = rc[0];
          int depth = rc[1];

          // retrieve the 4 coordinates that is used in the 4 packed output values.
          ivec4 coords = ivec4(rc.wz, rc.w + 1, rc.z + 1);

          // calculate the source index in fraction
          vec4 sourceFrac = getSourceFracIndex(coords);

          // get the lower and upper bound of the 4 values that will be packed into one texel.
          ivec4 x00 = ivec4(max(sourceFrac.xy, vec2(0.0)), min(inputWH - 1.0, ceil(sourceFrac.xy)));
          ivec4 x01 = ivec4(max(sourceFrac.xw, vec2(0.0)), min(inputWH - 1.0, ceil(sourceFrac.xw)));
          ivec4 x10 = ivec4(max(sourceFrac.zy, vec2(0.0)), min(inputWH - 1.0, ceil(sourceFrac.zy)));
          ivec4 x11 = ivec4(max(sourceFrac.zw, vec2(0.0)), min(inputWH - 1.0, ceil(sourceFrac.zw)));

          bool hasNextRow = rc.w < ${outputHeight - 1};
          bool hasNextCol = rc.z < ${outputWidth - 1};

          // pack x00, x01, x10, x11's top-left corner into one vec4 structure
          vec4 topLeft = vec4(
            getAValue(batch, depth, x00.x, x00.y),
            hasNextCol ? getAValue(batch, depth, x01.x, x01.y)
                      : 0.0,
            hasNextRow ? getAValue(batch, depth, x10.x, x10.y)
                      : 0.0,
            (hasNextRow && hasNextCol) ?
              getAValue(batch, depth, x11.x, x11.y) : 0.0);

          // pack x00, x01, x10, x11's top-right corner into one vec4 structure
          vec4 topRight = vec4(
            getAValue(batch, depth, x00.x, x00.w),
            hasNextCol ? getAValue(batch, depth, x01.x, x01.w)
                      : 0.0,
            hasNextRow ? getAValue(batch, depth, x10.x, x10.w)
                      : 0.0,
            (hasNextRow && hasNextCol) ?
              getAValue(batch, depth, x11.x, x11.w) : 0.0);

          // pack x00, x01, x10, x11's bottom-left corner into one vec4 structure
          vec4 bottomLeft = vec4(
            getAValue(batch, depth, x00.z, x00.y),
            hasNextCol ? getAValue(batch, depth, x01.z, x01.y)
                      : 0.0,
            hasNextRow ? getAValue(batch, depth, x10.z, x10.y)
                      : 0.0,
            (hasNextRow && hasNextCol) ?
              getAValue(batch, depth, x11.z, x11.y) : 0.0);

          // pack x00, x01, x10, x11's bottom-right corner into one vec4 structure
          vec4 bottomRight = vec4(
            getAValue(batch, depth, x00.z, x00.w),
            hasNextCol ? getAValue(batch, depth, x01.z, x01.w)
                      : 0.0,
            hasNextRow ? getAValue(batch, depth, x10.z, x10.w)
                      : 0.0,
            (hasNextRow && hasNextCol) ?
              getAValue(batch, depth, x11.z, x11.w) : 0.0);

          // calculate the interpolation fraction on u and v direction
          vec4 frac = vec4(sourceFrac) - floor(sourceFrac);
          vec4 clampFrac = clamp(frac, vec4(0.0), vec4(1.0));

          vec4 top = mix(topLeft, topRight, clampFrac.ywyw);
          vec4 bottom = mix(bottomLeft, bottomRight, clampFrac.ywyw);
          vec4 newValue = mix(top, bottom, clampFrac.xxzz);

          ${glsl.output} = vec4(newValue);
        }
      `;
  return {
    inputLayouts: [inputLayout],
    outputLayout,
    samplers: ['A'],
    shaderSource: shader,
    hasMain: true,
    expectPackedInputs: true,
    expectPackedOutputs: true,
  };
}
