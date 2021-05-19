// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {Reshape} from '../../../ops/reshape';
import {Tensor} from '../../../tensor';
import {ShapeUtil} from '../../../util';
import {getGlsl} from '../glsl-source';
import {WebGLInferenceHandler} from '../inference-handler';
import {ProgramInfo, RunData, TextureData, WebGLOperator} from '../types';
import {TextureLayout} from '../types';
import {unpackFromChannel} from './packing-utils';

export class WebGLReshapePacked extends Reshape implements WebGLOperator {
  run(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): Tensor[] {
    return inferenceHandler.run(this, inputs);
  }
  createProgramInfo(handler: WebGLInferenceHandler, inputs: Tensor[]): ProgramInfo {
    if (inputs.length !== 2) {
      throw new Error('resize kernel should have input tensor count to 2.');
    }

    // For packed reshape, we need to re-arrange texel data for output shape.
    // Our pack is designed to pack a 2x2 tile in last h and w dimension, so
    // for the reshaped new tensor, we just need to re-arrange the last h and
    // w dimension. For any shape that is not in 3D, i.e. [batch, W, H], we
    // first convert it to 3D by collapsing other dimension to batch dim, then
    // process with the last two dimensions.
    // Note: we only need the shape tensor to calculate output shape, so the
    // content in shape tensor is never uploaded to GPU. It is always kept in CPU.
    // TODO: optimize the algorithm -- in some cases, if the last two dims are
    // the same between input shape and output shape, the packed reshape can be
    // treated as no-op.
    // TODO: the implementation is a bit complicated due to the fact tensor shape is
    // immutable once the tensor is created, plus the tensor shape has a 1-to-1
    // mapping with texture layout. In the future, we may consider relaxing this
    // assumption.

    const originInputShape = inputs[0].dims;
    this.inputShape3D = processDims3D(inputs[0].dims);
    let inputLayout: TextureLayout;
    inputLayout = handler.getOrCreateTextureLayout(inputs[0], 4, true, originInputShape, true);
    if (originInputShape.length !== 3) {
      const originalInputLayout = inputLayout;
      // if originShape is not a 3D shape, create texture layout from the processed shape.
      inputLayout = handler.createTextureLayoutFromShape(
          this.inputShape3D, 4, this.inputShape3D, {isPacked: true, reverseWH: true});
      // if the processed input shape produces texture layout differnt from the original
      // one, the run data has to use the processed (3D) input shape later.
      this.needSqueezeInputData =
          (inputLayout.height !== originalInputLayout.height) || (inputLayout.width !== originalInputLayout.width);
    }

    this.outputShape = ShapeUtil.calculateReshapedDims(originInputShape, inputs[1].integerData);
    const squeezedOutputShape = processDims3D(this.outputShape);

    this.outputLayout = handler.createTextureLayoutFromShape(
        squeezedOutputShape, 4, squeezedOutputShape, {isPacked: true, reverseWH: true});

    let mainLoop = '';
    for (let i = 0; i < 4; i++) {
      let outputCoords = '';
      switch (i) {
        case 0:
          outputCoords = 'outputCoords = rc;';
          break;
        case 1:
          outputCoords = 'outputCoords = ivec3(rc.x, rc.y+1, rc.z);';
          break;
        case 2:
          outputCoords = 'outputCoords = ivec3(rc.x, rc.y, rc.z+1);';
          break;
        case 3:
          outputCoords = 'outputCoords = ivec3(rc.x, rc.y+1, rc.z+1);';
          break;
        default:
          throw new Error();
      }

      mainLoop += `
        ${outputCoords}
        ${i > 0 ? 'if(outputCoords.y < rows && outputCoords.z < cols){' : ''}
          int flattenedIndex = getFlattenedIndex(outputCoords);

          ivec3 inputRC = inputCoordsFromReshapedOutCoords(flattenedIndex);
          vec2 innerDims = vec2(float(inputRC.y),float(inputRC.z));

          result[${i}] = getChannel(getA(inputRC.x, inputRC.y, inputRC.z), innerDims);

        ${i > 0 ? '}' : ''}
      `;
    }
    const glsl = getGlsl(handler.session.backend.glContext.version);

    const shaderSource = `
      ${getReshapedInputCoords(this.inputShape3D)}
      ${getFlattenedIndexFrom3D(squeezedOutputShape)}
      ${unpackFromChannel()}

      void main() {
        ivec3 rc = getOutputCoords();

        vec4 result = vec4(0.0);

        ivec3 outputCoords;
        int rows = ${squeezedOutputShape[2]};
        int cols = ${squeezedOutputShape[1]};

        ${mainLoop}
        ${glsl.output} = result;
      }
    `;

    return {
      name: 'WebGLReshapePacked',
      inputLayouts: [inputLayout],
      outputLayout: this.outputLayout,
      samplers: ['A'],
      shaderSource,
      hasMain: true,
      expectPackedInputs: true,
      expectPackedOutputs: true,
    };
  }
  createRunData(handler: WebGLInferenceHandler, programInfo: ProgramInfo, inputs: Tensor[]): RunData {
    let inputTDs: [TextureData];
    const originalInputLayout = handler.getOrCreateTextureLayout(inputs[0], 1, false, [], false);
    const originalInputTD = handler.getOrCreateTextureData(inputs[0], originalInputLayout, false);

    if (this.needSqueezeInputData) {
      const squeezedInputLayout: TextureLayout = {
        channels: 1,
        height: originalInputLayout.height,
        width: originalInputLayout.width,
        shape: this.inputShape3D,
        strides: ShapeUtil.computeStrides(this.inputShape3D),
        unpackedShape: this.inputShape3D,
      };
      const squeezedInputTD =
          handler.createSharedTextureData(squeezedInputLayout, inputs[0].type, originalInputTD.texture);
      inputTDs = [squeezedInputTD];

    } else {
      inputTDs = [originalInputTD];
    }
    let outputLayout = this.outputLayout;
    // Use original output shape for runData output layout.
    outputLayout.shape = this.outputShape;
    outputLayout.unpackedShape = this.outputShape;
    if (outputLayout === undefined) {
      outputLayout = handler.createTextureLayoutFromShape(
          this.outputShape, 4, this.outputShape, {isPacked: true, reverseWH: true});
    }
    // return run data for reshape. Here, we use the original calculate outputLayout to create the real output layout.
    return {
      inputTextureDatas: inputTDs,
      outputTextureData: handler.createTextureDataFromLayout(outputLayout, inputTDs[0].tensor.type),
      uniformData: {}
    };
  }
  protected outputShape: readonly number[];
  private inputShape3D: [number, number, number];
  private needSqueezeInputData = false;
  private outputLayout: TextureLayout;
}

function processDims3D(shape: readonly number[]|readonly number[]|Tensor.IntegerType): [number, number, number] {
  if (shape.length === 0) {
    return [1, 1, 1];
  }
  // TODO: squeeze other shapes to 2D case
  const batchDims = shape.length >= 3 ? shape.slice(0, shape.length - 2) : [1];
  let batch = 1;
  for (let i = 0; i < batchDims.length; ++i) {
    batch *= batchDims[i];
  }
  return [batch, shape.length > 1 ? shape[shape.length - 2] : 1, shape[shape.length - 1]];
}
function getReshapedInputCoords(shape: [number, number, number]): string {
  const strides = ShapeUtil.computeStrides(shape);
  const coords = ['b', 'r', 'c'];
  const index = 'index';
  const coordsFromIndexSnippet = strides
                                     .map((stride, i) => {
                                       const line1 = `int ${coords[i]} = ${index} / ${stride}`;
                                       const line2 = i === strides.length - 1 ?
                                           `int ${coords[i + 1]} = ${index} - ${coords[i]} * ${stride}` :
                                           `index -= ${coords[i]} * ${stride}`;
                                       return `${line1}; ${line2};`;
                                     })
                                     .join('');

  return `
    ivec3 inputCoordsFromReshapedOutCoords(int index) {
      ${coordsFromIndexSnippet}
      return ivec3(b, r, c);
    }
  `;
}

function getFlattenedIndexFrom3D(shape: [number, number, number]): string {
  const strides = ShapeUtil.computeStrides(shape);

  return `
  int getFlattenedIndex(ivec3 coords) {
    // reverse y, z order
    return coords.x * ${strides[0]} + coords.z * ${strides[1]} + coords.y;
  }
`;
}
