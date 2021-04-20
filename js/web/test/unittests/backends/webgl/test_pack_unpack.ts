// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.
import {expect} from 'chai';

import {Backend, InferenceHandler, resolveBackend, SessionHandler} from '../../../../lib/onnxjs/backend';
import {WebGLInferenceHandler} from '../../../../lib/onnxjs/backends/webgl/inference-handler';
import {WebGLPack} from '../../../../lib/onnxjs/backends/webgl/ops/pack';
import {WebGLUnpack} from '../../../../lib/onnxjs/backends/webgl/ops/unpack';
import {WebGLContext} from '../../../../lib/onnxjs/backends/webgl/webgl-context';
import {Profiler} from '../../../../lib/onnxjs/instrument';
import {Tensor} from '../../../../lib/onnxjs/tensor';
import {ShapeUtil} from '../../../../lib/onnxjs/util';

let backend: Backend|undefined;
let sessionhandler: SessionHandler|undefined;
let inferenceHandler: InferenceHandler|undefined;

function createAscendingArray(size: number): Float32Array {
  return new Float32Array(Array.from({length: size}, (v, i) => (i + 1)));
}

// create a webgl texture and fill it with the array content
function createTextureFromArray(
    glContext: WebGLContext, dataArray: Float32Array, type: GLenum, width: number, height: number) {
  const gl = glContext.gl;

  // create the texture
  const texture = gl.createTexture();
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
function createArrayFromTexture(
    gl: WebGLRenderingContext, texture: WebGLTexture, width: number, height: number): Float32Array {
  const resultDataBuffer = new Float32Array(width * height * 4);
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.framebufferTexture2D(
      gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture,
      0);  // 0, we aren't using MIPMAPs
  gl.readPixels(0, 0, width, height, gl.RGBA, gl.FLOAT, resultDataBuffer);
  return resultDataBuffer;
}

function getExpectedElementCount(inputShape: number[], isPacked = true): number {
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

function generateExpected(inputArray: Float32Array, inputShape: number[]): Float32Array {
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

interface TestData {
  elementCount: number;
  inputShape: number[];
  outputShape: number[];
  inputTextureShape: number[];
  outputTextureShape: number[];
  rawData?: Float32Array;
  useGeneratedOutput?: boolean;
}
function getTestData(isPacked = true): TestData[] {
  if (isPacked) {
    return [
      // test scalar
      {elementCount: 1, inputShape: [], outputShape: [], inputTextureShape: [], outputTextureShape: [1, 1]},

      // test 1D tensor
      {elementCount: 1, inputShape: [1], outputShape: [], inputTextureShape: [], outputTextureShape: [1, 1]},
      {elementCount: 16, inputShape: [16], outputShape: [], inputTextureShape: [], outputTextureShape: [1, 8]},
      {elementCount: 9, inputShape: [9], outputShape: [], inputTextureShape: [], outputTextureShape: [1, 5]},

      // test 2D tensor
      {elementCount: 1, inputShape: [1, 1], outputShape: [], inputTextureShape: [], outputTextureShape: [1, 1]},
      {elementCount: 16, inputShape: [4, 4], outputShape: [], inputTextureShape: [], outputTextureShape: [2, 2]},
      {elementCount: 16, inputShape: [2, 8], outputShape: [], inputTextureShape: [], outputTextureShape: [1, 4]},
      {elementCount: 16, inputShape: [8, 2], outputShape: [], inputTextureShape: [], outputTextureShape: [4, 1]},
      {elementCount: 15, inputShape: [3, 5], outputShape: [], inputTextureShape: [], outputTextureShape: [2, 3]},
      {elementCount: 18, inputShape: [3, 6], outputShape: [], inputTextureShape: [], outputTextureShape: [2, 3]},
      {elementCount: 10, inputShape: [2, 5], outputShape: [], inputTextureShape: [], outputTextureShape: [1, 3]},
      {elementCount: 6, inputShape: [1, 6], outputShape: [], inputTextureShape: [], outputTextureShape: [1, 3]},
      {elementCount: 6, inputShape: [6, 1], outputShape: [], inputTextureShape: [], outputTextureShape: [3, 1]},
      {elementCount: 5, inputShape: [5, 1], outputShape: [], inputTextureShape: [], outputTextureShape: [3, 1]},
      {elementCount: 5, inputShape: [1, 5], outputShape: [], inputTextureShape: [], outputTextureShape: [1, 3]},

      // test 3D tensor
      {elementCount: 1, inputShape: [1, 1, 1], outputShape: [], inputTextureShape: [], outputTextureShape: [1, 1]},
      {elementCount: 16, inputShape: [2, 2, 4], outputShape: [], inputTextureShape: [], outputTextureShape: [2, 2]},
      {elementCount: 24, inputShape: [2, 3, 4], outputShape: [], inputTextureShape: [], outputTextureShape: [4, 2]},
      {elementCount: 30, inputShape: [5, 3, 2], outputShape: [], inputTextureShape: [], outputTextureShape: [10, 1]},
      {elementCount: 9, inputShape: [1, 3, 3], outputShape: [], inputTextureShape: [], outputTextureShape: [2, 2]},

      // test 4D tensor
      {elementCount: 1, inputShape: [1, 1, 1, 1], outputShape: [], inputTextureShape: [], outputTextureShape: [1, 1]},
      {elementCount: 16, inputShape: [1, 2, 2, 4], outputShape: [], inputTextureShape: [], outputTextureShape: [2, 2]},
      {elementCount: 32, inputShape: [2, 2, 2, 4], outputShape: [], inputTextureShape: [], outputTextureShape: [4, 2]},
      {elementCount: 36, inputShape: [2, 2, 3, 3], outputShape: [], inputTextureShape: [], outputTextureShape: [8, 2]},
      {elementCount: 80, inputShape: [2, 5, 2, 4], outputShape: [], inputTextureShape: [], outputTextureShape: [10, 2]},
      {elementCount: 12, inputShape: [2, 1, 3, 2], outputShape: [], inputTextureShape: [], outputTextureShape: [4, 1]},
    ];
  } else {
    return [
      // // test 1D tensor
      {
        elementCount: 8,
        inputShape: [8],
        outputShape: [8],
        inputTextureShape: [4, 1],
        outputTextureShape: [1, 8],
        rawData: new Float32Array([1, 2, 0, 0, 3, 4, 0, 0, 5, 6, 0, 0, 7, 8, 0, 0]),
      },

      // // test 2D tensor
      {
        elementCount: 16,
        inputShape: [4, 4],
        outputShape: [4, 4],
        inputTextureShape: [2, 2],
        outputTextureShape: [4, 4],
        useGeneratedOutput: true,
      },
      {
        elementCount: 8,
        inputShape: [2, 4],
        outputShape: [2, 4],
        inputTextureShape: [2, 1],
        outputTextureShape: [2, 4],
        useGeneratedOutput: true,
      },
      {
        elementCount: 6,
        inputShape: [2, 3],
        outputShape: [2, 3],
        inputTextureShape: [2, 1],
        outputTextureShape: [2, 3],
        rawData: new Float32Array([1, 2, 4, 5, 3, 0, 6, 0]),
      },

      // // test 3d tensor
      {
        elementCount: 16,
        inputShape: [2, 2, 4],
        outputShape: [2, 2, 4],
        inputTextureShape: [2, 2],
        outputTextureShape: [4, 4],
        useGeneratedOutput: true,
      },
      {
        elementCount: 24,
        inputShape: [2, 3, 4],
        outputShape: [2, 3, 4],
        inputTextureShape: [2, 4],
        outputTextureShape: [6, 4],
        rawData: new Float32Array([
          1,  2,  5,  6,  3,  4,  7,  8,  9,  10, 0, 0, 11, 12, 0, 0,
          13, 14, 17, 18, 15, 16, 19, 20, 21, 22, 0, 0, 23, 24, 0, 0
        ])
      },
      // test 4d tensor
      {
        elementCount: 32,
        inputShape: [2, 2, 2, 4],
        outputShape: [2, 2, 2, 4],
        inputTextureShape: [2, 4],
        outputTextureShape: [8, 4],
        useGeneratedOutput: true,
      },
      {
        elementCount: 64,
        inputShape: [2, 2, 4, 4],
        outputShape: [2, 2, 4, 4],
        inputTextureShape: [2, 8],
        outputTextureShape: [16, 4],
        useGeneratedOutput: true,
      },
    ];
  }
}

describe('#UnitTest# - pack - Tensor pack', () => {
  before('Initialize Context', async () => {
    const profiler = Profiler.create();
    backend = await resolveBackend('webgl');
    sessionhandler = backend.createSessionHandler({profiler});
    inferenceHandler = sessionhandler.createInferenceHandler();
  });
  const testDataSet = getTestData();
  for (let k = 0; k < testDataSet.length; ++k) {
    const testData = testDataSet[k];
    describe(`Test pack ${JSON.stringify(testData)}`, () => {});
    it(`Test pack kernal ${JSON.stringify(testData)}`, () => {
      const op = new WebGLPack();

      const elementCount = testData.elementCount;
      const inputData = createAscendingArray(elementCount);
      const inputTensorShape = testData.inputShape;
      const outputTextureShape = testData.outputTextureShape;

      const inputTensor = new Tensor(inputTensorShape, 'float32', undefined, undefined, inputData);

      // compile shader code
      const programInfo = op.createProgramInfo(inferenceHandler! as WebGLInferenceHandler, [inputTensor]);
      const webglInferenceHandler = inferenceHandler as WebGLInferenceHandler;
      const artifact = webglInferenceHandler.session.programManager.build(programInfo);
      webglInferenceHandler.session.programManager.setArtifact(op, artifact);

      // run kernal and get output
      const runData = op.createRunData(webglInferenceHandler, artifact.programInfo, [inputTensor]);
      webglInferenceHandler.session.programManager.run(artifact, runData);
      const resultTexture = runData.outputTextureData.texture;
      const gl = webglInferenceHandler.session.textureManager.glContext.gl;
      const resultDataBuffer = createArrayFromTexture(gl, resultTexture, outputTextureShape[1], outputTextureShape[0]);

      expect(resultDataBuffer).to.not.equal(null);

      const outputElementCount = getExpectedElementCount(testData.inputShape);
      expect(resultDataBuffer).to.have.lengthOf(outputElementCount);
      console.log(resultDataBuffer);
      const expectedOutput = generateExpected(inputData, testData.inputShape);
      expect(resultDataBuffer).to.deep.equal(expectedOutput);
    });
  }
});

describe('#UnitTest# - unpack - Tensor unpack', () => {
  before('Initialize Context', async () => {
    const profiler = Profiler.create();
    backend = await resolveBackend('webgl');
    sessionhandler = backend.createSessionHandler({profiler});
    inferenceHandler = sessionhandler.createInferenceHandler();
  });
  const testDataSet = getTestData(false);
  for (let k = 0; k < testDataSet.length; ++k) {
    const testData = testDataSet[k];
    describe(`Test unpack ${JSON.stringify(testData)}`, () => {});
    it('Test unpack kernal ', () => {
      const op = new WebGLUnpack();

      const elementCount = testData.elementCount;
      const inputTensorShape = testData.inputShape;
      const inputTextureShape = testData.inputTextureShape;
      const outputTensorShape = testData.outputShape;

      // create input data and tensor. The input data will be used to verify if the output tensor contains the
      // same value but possibly different order depending on our packing algorithm.
      const inputData = createAscendingArray(elementCount);
      const inputTensor = new Tensor(inputTensorShape, 'float32', undefined, undefined, inputData);

      const webglInferenceHandler = inferenceHandler as WebGLInferenceHandler;

      // manually creat packed texture from inputTensor, and insert in cache
      const gl = webglInferenceHandler.session.textureManager.glContext.gl;
      webglInferenceHandler.session.textureManager.glContext.checkError();
      const webglTexture = createTextureFromArray(
          webglInferenceHandler.session.textureManager.glContext, testData.rawData ? testData.rawData : inputData,
          gl.RGBA, inputTextureShape[0], inputTextureShape[1]);
      webglInferenceHandler.session.textureManager.glContext.checkError();
      const packedShape = inputTextureShape;
      const textureData = {
        width: inputTextureShape[0],
        height: inputTextureShape[1],
        channels: 4 as const,
        isPacked: true,
        shape: packedShape,
        strides: ShapeUtil.computeStrides(packedShape),
        unpackedShape: outputTensorShape,
        tensor: inputTensor,
        texture: webglTexture!
      };

      webglInferenceHandler.setTextureData(inputTensor.dataId, textureData);

      // compile shader code
      const programInfo = op.createProgramInfo(inferenceHandler! as WebGLInferenceHandler, [inputTensor]);

      const artifact = webglInferenceHandler.session.programManager.build(programInfo);
      webglInferenceHandler.session.programManager.setArtifact(op, artifact);

      // run kernal and get output
      const runData = op.createRunData(webglInferenceHandler, artifact.programInfo, [inputTensor]);
      webglInferenceHandler.session.programManager.run(artifact, runData);
      const result = runData.outputTextureData.tensor.data;

      const resultDataBuffer = createArrayFromTexture(gl, webglTexture!, inputTextureShape[0], inputTextureShape[1]);

      webglInferenceHandler.session.textureManager.glContext.checkError();
      // verify result.
      const expectedOutput = testData.useGeneratedOutput ? generateExpected(inputData, testData.inputShape) : inputData;
      expect(result).to.not.equal(null);
      expect(result).to.have.lengthOf(elementCount);

      expect(resultDataBuffer).to.deep.equal(testData.rawData ? testData.rawData : inputData);
      const outputElementCount = getExpectedElementCount(testData.inputShape);

      expect(resultDataBuffer).to.have.lengthOf(outputElementCount);
      expect(result).to.deep.equal(expectedOutput);
    });
  }
});
