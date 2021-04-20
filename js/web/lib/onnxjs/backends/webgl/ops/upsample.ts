// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Upsample} from '../../../ops/upsample';
import {Tensor} from '../../../tensor';
import {getGlsl} from '../glsl-source';
import {WebGLInferenceHandler} from '../inference-handler';
import {ProgramInfo, RunData, WebGLOperator} from '../types';

export class WebGLUpsample extends Upsample implements WebGLOperator {
  run(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): Tensor[] {
    return inferenceHandler.run(this, inputs);
  }
  createProgramInfo(handler: WebGLInferenceHandler, inputs: Tensor[]): ProgramInfo {
    const inputLayout = handler.getOrCreateTextureLayout(inputs[0]);
    const outputShape = inputs[0].dims.map((dim, i) => Math.floor(dim * this.scales[i]));
    const outputLayout = handler.createTextureLayoutFromShape(outputShape);
    const dim = outputShape.length;

    const glsl = getGlsl(handler.session.backend.glContext.version);

    const outputPitches = new Array<number>(dim);
    const inputPitches = new Array<number>(dim);
    let precalculatedPitches = `
      int output_pitches[${dim}];
      int input_pitches[${dim}];
      `;
    for (let d = dim - 1; d >= 0; d--) {
      outputPitches[d] = (d === dim - 1) ? 1 : outputPitches[d + 1] * outputShape[d + 1];
      inputPitches[d] = (d === dim - 1) ? 1 : inputPitches[d + 1] * inputs[0].dims[d + 1];

      precalculatedPitches += `
      output_pitches[${d}] = ${outputPitches[d]};
      input_pitches[${d}] = ${inputPitches[d]};
      `;
    }
    const getInputFloatFunction = `
    float getInputFloat(int index) {
      vec2 coords = offsetToCoords(index, ${inputLayout.width}, ${inputLayout.height});
      float value = getColorAsFloat(${glsl.texture2D}(X, coords));
      return value;
    }
    `;

    const shaderSource = this.mode === 'nearest' ?
        // nearest
        `
      ${getInputFloatFunction}
      float process(int indices[${dim}]) {
        int input_index = 0;
        int output_index = coordsToOffset(TexCoords, ${outputLayout.width}, ${outputLayout.height});

        ${precalculatedPitches}

        int d, m;
        for (int dim = 0; dim < ${dim}; ++dim) {
          d = output_index / output_pitches[dim];
          m = output_index - d * output_pitches[dim];
          output_index = m;

          if (scales[dim] != 1 && d > 0) {
            int d2 = d / scales[dim];
            m = d - d2 * scales[dim];
            d = d2;
          }
          input_index += input_pitches[dim] * d;
        }

        return getInputFloat(input_index);
      }` :
        dim === 4 ?
        // bilinear 4D
            `
      ${getInputFloatFunction}
      float process(int indices[4]) {
        int input_index = 0;
        int output_index = coordsToOffset(TexCoords, ${outputLayout.width}, ${outputLayout.height});

        ${precalculatedPitches}

        int m;
        int index_of_dim0, index_of_dim1, index_of_dim2, index_of_dim3;
        index_of_dim0 = output_index / output_pitches[0];
        m = output_index - index_of_dim0 * output_pitches[0];
        index_of_dim1 = m / output_pitches[1];
        m = m - index_of_dim1 * output_pitches[1];
        index_of_dim2 = m / output_pitches[2];
        m = m - index_of_dim2 * output_pitches[2];
        index_of_dim3 = m;

        int index_of_input_dim2, index_of_input_dim3, x_offset, y_offset;
        index_of_input_dim2 = index_of_dim2 / scales[2];
        y_offset = index_of_dim2 - index_of_input_dim2 * scales[2];
        index_of_input_dim3 = index_of_dim3 / scales[3];
        x_offset = index_of_dim3 - index_of_input_dim3 * scales[3];

        input_index = index_of_dim0 * input_pitches[0] +
                      index_of_dim1 * input_pitches[1] +
                      index_of_input_dim2 * input_pitches[2] +
                      index_of_input_dim3;

        float x00 = getInputFloat(input_index);
        float x10, x01, x11;

        bool end_of_dim2 = false;
        if (index_of_input_dim2 == (${inputs[0].dims[2]} - 1)) {
          // It's the end in dimension 2
          x01 = x00;
          end_of_dim2 = true;
        } else {
          x01 = getInputFloat(input_index + input_pitches[2]);
        }

        if (index_of_input_dim3 == (input_pitches[2] - 1)) {
          // It's the end in dimension 3
          x10 = x00;
          x11 = x01;
        }
        else {
          x10 = getInputFloat(input_index + 1);
          x11 = end_of_dim2 ? x10 : getInputFloat(input_index + input_pitches[2] + 1);
        }

        float y0 = x00 + float(y_offset) * (x01 - x00) / float(scales[2]);
        float y1 = x10 + float(y_offset) * (x11 - x10) / float(scales[2]);
        return y0 + float(x_offset) * (y1 - y0) / float(scales[3]);
      }` :
            // bilinear 2D
            `
      ${getInputFloatFunction}
      float process(int indices[2]) {
        int input_index = 0;
        int output_index = coordsToOffset(TexCoords, ${outputLayout.width}, ${outputLayout.height});

        ${precalculatedPitches}

        int m;
        int index_of_dim0, index_of_dim1;
        index_of_dim0 = output_index / output_pitches[0];
        m = output_index - index_of_dim0 * output_pitches[0];
        index_of_dim1 = m;

        int index_of_input_dim0, index_of_input_dim1, x_offset, y_offset;
        index_of_input_dim0 = index_of_dim0 / scales[0];
        y_offset = index_of_dim0 - index_of_input_dim0 * scales[0];
        index_of_input_dim1 = index_of_dim1 / scales[1];
        x_offset = index_of_dim1 - index_of_input_dim1 * scales[1];

        input_index = index_of_input_dim0 * input_pitches[0] + index_of_input_dim1;

        float x00 = getInputFloat(input_index);
        float x10, x01, x11;

        bool end_of_dim0 = false;
        if (index_of_input_dim0 == (${inputs[0].dims[0]} - 1)) {
          // It's the end in dimension 0
          x01 = x00;
          end_of_dim0 = true;
        } else {
          x01 = getInputFloat(input_index + input_pitches[0]);
        }

        if (index_of_input_dim1 == (input_pitches[0] - 1)) {
          // It's the end in dimension 1
          x10 = x00;
          x11 = x01;
        }
        else {
          x10 = getInputFloat(input_index + 1);
          x11 = end_of_dim0 ? x10 : getInputFloat(input_index + input_pitches[0] + 1);
        }

        float y0 = x00 + float(y_offset) * (x01 - x00) / float(scales[0]);
        float y1 = x10 + float(y_offset) * (x11 - x10) / float(scales[0]);
        return y0 + float(x_offset) * (y1 - y0) / float(scales[1]);
      }`;
    return {
      inputLayouts: [inputLayout],
      outputLayout,
      samplers: ['X'],
      shaderSource,
      variables: [{name: 'scales', type: 'int', arrayLength: this.scales.length}]
    };
  }
  createRunData(handler: WebGLInferenceHandler, programInfo: ProgramInfo, inputs: Tensor[]): RunData {
    const inputTDs = inputs.map((t, i) => handler.getOrCreateTextureData(t, programInfo.inputLayouts[i]));
    return {
      inputTextureDatas: inputTDs,
      outputTextureData: handler.createTextureDataFromLayout(programInfo.outputLayout, inputTDs[0].tensor.type),
      uniformData: {scales: this.scales.map(x => Math.ceil(x))}
    };
  }
}
