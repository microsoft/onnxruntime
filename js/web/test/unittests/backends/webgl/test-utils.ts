// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {WebGLContext} from '../../../../lib/onnxjs/backends/webgl/webgl-context';

export function createAscendingArray(size: number): Float32Array {
  return new Float32Array(Array.from({length: size}, (_v, i) => (i + 1)));
}

// Returns an array by injecting 3 zeros after every element in the input array to be used for creating unpacked
// texture.
export function generateArrayForUnpackedTexture(input: Float32Array): Float32Array {
  const output = new Float32Array(input.length * 4);
  for (let i = 0; i < (input.length * 4); i += 4) {
    output[i] = input[i / 4];
  }
  return output;
}

// create a webgl texture and fill it with the array content
export function createTextureFromArray(
    glContext: WebGLContext, dataArray: Float32Array, width: number, height: number): WebGLTexture {
  const gl = glContext.gl;

  // create the texture
  const texture = gl.createTexture();
  if (!texture) {
    throw new Error('failed to create texture');
  }
  // bind the texture so the following methods effect this texture.
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  if (glContext.version === 2) {
    const webgl2Gl = gl as WebGL2RenderingContext;
    gl.texImage2D(webgl2Gl.TEXTURE_2D, 0, webgl2Gl.RGBA32F, width, height, 0, webgl2Gl.RGBA, webgl2Gl.FLOAT, dataArray);
  } else {
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, width, height, 0, gl.RGBA, gl.FLOAT, dataArray);
  }

  glContext.checkError();
  return texture;
}

// create a cpu array and download GPU texture data to this array
export function createArrayFromTexture(
    gl: WebGLRenderingContext, texture: WebGLTexture, width: number, height: number): Float32Array {
  const resultDataBuffer = new Float32Array(width * height * 4);
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.framebufferTexture2D(
      gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture,
      0);  // 0, we aren't using MIPMAPs
  gl.readPixels(0, 0, width, height, gl.RGBA, gl.FLOAT, resultDataBuffer);
  return resultDataBuffer;
}

export function getExpectedElementCount(inputShape: number[], isPacked = true): number {
  const rank = inputShape.length;

  if (isPacked) {
    // scalar
    if (rank === 0) {
      return 4;
    }

    // 1D tensor
    if (rank === 1) {
      if (inputShape[0] % 2) {
        return (inputShape[0] + 1) * 2;
      } else {
        return inputShape[0] * 2;
      }
    }

    // process width
    let inputWidth = inputShape[rank - 2] % 2 ? inputShape[rank - 2] + 1 : inputShape[rank - 2];
    if (rank > 2) {
      for (let i = 0; i < rank - 2; ++i) {
        inputWidth *= inputShape[i];
      }
    }
    // process height
    let inputHeight = inputShape[rank - 1];
    if (inputHeight % 2) {
      inputHeight++;
    }
    return inputWidth * inputHeight;
  } else {
    let totalCount = 1;
    for (let i = 0; i < rank; i++) {
      totalCount *= inputShape[i];
    }
    return totalCount;
  }
}

export function generateExpected(inputArray: Float32Array, inputShape: number[]): Float32Array {
  if (inputShape.length === 0) {
    const result = new Float32Array(4);
    result[0] = inputArray[0];
    return result;
  }
  const rank = inputShape.length;

  const inputHeight = rank === 1 ? 1 : inputShape[rank - 2];
  const inputWidth = inputShape[rank - 1];
  const paddedW = inputWidth % 2 ? inputWidth + 1 : inputWidth;
  const paddedH = inputHeight % 2 ? inputHeight + 1 : inputHeight;

  let B = 1;
  if (rank > 2) {
    for (let i = 0; i < rank - 2; ++i) {
      B *= inputShape[i];
    }
  }

  const result = new Float32Array(B * paddedW * paddedH);
  let ii = 0;
  for (let b = 0; b < B; ++b) {
    for (let j = 0; j < paddedH; j += 2) {
      for (let i = 0; i < paddedW; i += 2) {
        const index = j * inputWidth + i + b * (inputHeight * inputWidth);
        result[ii++] = inputArray[index];

        if (i + 1 < inputWidth) {
          result[ii++] = inputArray[index + 1];
        } else {
          result[ii++] = 0;
        }

        if ((j + 1) < inputHeight) {
          result[ii++] = inputArray[(j + 1) * inputWidth + i + b * (inputHeight * inputWidth)];
        } else {
          result[ii++] = 0;
        }

        if (i + 1 < inputWidth && j + 1 < inputHeight) {
          result[ii++] = inputArray[(j + 1) * inputWidth + i + 1 + b * (inputHeight * inputWidth)];
        } else {
          result[ii++] = 0;
        }
      }
    }
  }
  return result;
}
