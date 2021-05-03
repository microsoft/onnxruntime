// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {Concat} from '../../../ops/concat';
import {Tensor} from '../../../tensor';
import {getGlsl} from '../glsl-source';
import {WebGLInferenceHandler} from '../inference-handler';
import {ProgramInfo, RunData, WebGLOperator} from '../types';
import {getCoordsDataType} from '../utils';

import {getChannels, unpackFromChannel} from './packing-utils';

export class WebGLPackedConcat extends Concat implements WebGLOperator {
  run(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): Tensor[] {
    return inferenceHandler.run(this, inputs);
  }
  createProgramInfo(handler: WebGLInferenceHandler, inputs: Tensor[]): ProgramInfo {
    const inputShape = inputs[0].dims.slice();
    if (this.axis >= inputShape.length || this.axis < (-1 * inputShape.length)) {
      throw new Error('axis specified for concat doesn\'t match input dimensionality');
    }
    if (this.axis < 0) {
      this.axis = inputShape.length + this.axis;
    }
    // ensure all of the non-concatenated axes match each other
    // calculate the shape of the output tensor while we do that
    const outputShape = inputShape.slice(0);
    for (let i = 1; i < inputs.length; i++) {
      const dataNShape = inputs[i].dims.slice();
      for (let axisIndex = 0; axisIndex < inputShape.length; axisIndex++) {
        // add to the placeholder for computing output shape
        if (axisIndex === this.axis) {
          outputShape[this.axis] += dataNShape[axisIndex];
        }
        // ensure all non-cancatenated axes match each other
        else if (inputShape[axisIndex] !== dataNShape[axisIndex]) {
          throw new Error('non concat dimensions must match');
        }
      }
    }

    const rank = outputShape.length;
    const coords = getChannels('coords', rank);
    const dtype = getCoordsDataType(rank);
    const unpackChannel = unpackFromChannel();

    const shapes = inputs.map(i => i.dims);
    const channels = ['x', 'y', 'z', 'w', 'u', 'v'].slice(0, rank);
    const offsets: number[] = new Array(shapes.length - 1);
    const samplers = inputs.map((v, i) => `X${i}`);

    offsets[0] = shapes[0][this.axis];
    for (let i = 1; i < offsets.length; i++) {
      offsets[i] = offsets[i - 1] + shapes[i][this.axis];
    }

    const channel = channels[this.axis];
    const lastChannels = channels.slice(-2);
    const allChannels = channels.join();

    let getValueSnippet = `if (${channel} < ${offsets[0]}) {
      return getChannel(
          getX0(${allChannels}), vec2(${lastChannels.join()}));
      }`;
    for (let i = 1; i < offsets.length; i++) {
      const shift = offsets[i - 1];
      getValueSnippet += `
          if (${channel} < ${offsets[i]}  && ${channel} >= ${offsets[i - 1]}) {
            return getChannel(
              getX${i}(${this.getShiftedChannelsSnippet(channels, channel, shift)}),
              vec2(${this.getShiftedChannelsSnippet(lastChannels, channel, shift)}));
          }`;
    }
    const lastIndex = offsets.length;
    const shift = offsets[offsets.length - 1];
    getValueSnippet += `
          return getChannel(
            getX${lastIndex}(${this.getShiftedChannelsSnippet(channels, channel, shift)}),
            vec2(${this.getShiftedChannelsSnippet(lastChannels, channel, shift)}));`;

    const glsl = getGlsl(handler.session.backend.glContext.version);

    const shaderSource = `
        ${unpackChannel}
        float getValue(${channels.map(x => 'int ' + x)}) {
          ${getValueSnippet}
        }

        void main() {
          ${dtype} coords = getOutputCoords();
          vec4 result = vec4(getValue(${coords}), 0., 0., 0.);

          ${coords[rank - 1]} = ${coords[rank - 1]} + 1;
          if (${coords[rank - 1]} < ${outputShape[rank - 1]}) {
            result.g = getValue(${coords});
          }

          ${coords[rank - 2]} = ${coords[rank - 2]} + 1;
          if (${coords[rank - 2]} < ${outputShape[rank - 2]}) {
            result.a = getValue(${coords});
          }

          ${coords[rank - 1]} = ${coords[rank - 1]} - 1;
          if (${coords[rank - 2]} < ${outputShape[rank - 2]} &&
              ${coords[rank - 1]} < ${outputShape[rank - 1]}) {
            result.b = getValue(${coords});
          }
          ${glsl.output} = result;
        }
      `;

    return {
      inputLayouts: inputs.map(t => handler.getOrCreateTextureLayout(t)),
      outputLayout: handler.createTextureLayoutFromShape(outputShape),
      samplers,
      shaderSource,
      hasMain: true,
      expectPackedInputs: true,
      expectPackedOutputs: true
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

  /**
   * Generates the snippet to shift a given channel in a list of channels by shift
   *
   * i.e: returns a string of the form 'x, y-[shift], z' where any one channel can
   * have the shift applied.
   */
  protected getShiftedChannelsSnippet(channels: string[], channel: string, shift: number) {
    const channelIdx = channels.indexOf(channel);
    const res = channels.map((c, idx) => {
      if (idx === channelIdx) {
        return `${c} - ${shift}`;
      } else {
        return c;
      }
    });
    return res.join();
  }
}
