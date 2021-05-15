// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {expect} from 'chai';

import {Attribute} from '../../../../lib/onnxjs/attribute';
import {Backend, InferenceHandler, resolveBackend, SessionHandler} from '../../../../lib/onnxjs/backend';
import {WebGLBackend} from '../../../../lib/onnxjs/backends/backend-webgl';
import {WebGLInferenceHandler} from '../../../../lib/onnxjs/backends/webgl/inference-handler';
import {WebGLConcat} from '../../../../lib/onnxjs/backends/webgl/ops/concat';
import {Profiler} from '../../../../lib/onnxjs/instrument';
import {Tensor} from '../../../../lib/onnxjs/tensor';
import {ShapeUtil} from '../../../../lib/onnxjs/util';

import {createAscendingArray} from './test-utils';
import {createTextureFromArray} from './test-utils';

interface TestData {
  elementCount: number;
  axis: number;
  inputShape: number[];
  outputShape: number[];
  inputTextureShape: number[];
  outputTextureShape: number[];
  expectedOutput: Float32Array;
  // If empty, the test will use auto-generated data.
  rawInput?: Float32Array;
}
function getTestData(): TestData[] {
  return [
    // test 2D tensor
    {
      elementCount: 16,
      axis: 0,
      inputShape: [4, 4],
      outputShape: [8, 4],
      inputTextureShape: [2, 2],
      outputTextureShape: [2, 4],
      expectedOutput: new Float32Array([
        1, 2, 5, 6, 3, 4, 7, 8, 9, 10, 13, 14, 11, 12, 15, 16, 1, 2, 5, 6, 3, 4, 7, 8, 9, 10, 13, 14, 11, 12, 15, 16
      ]),
    },
    {
      elementCount: 16,
      axis: 1,
      inputShape: [4, 4],
      outputShape: [4, 8],
      inputTextureShape: [2, 2],
      outputTextureShape: [4, 2],
      expectedOutput: new Float32Array([
        1, 2, 5, 6, 1, 2, 5, 6, 3, 4, 7, 8, 3, 4, 7, 8, 9, 10, 13, 14, 9, 10, 13, 14, 11, 12, 15, 16, 11, 12, 15, 16
      ]),
    },
    {
      elementCount: 8,
      axis: 0,
      inputShape: [2, 4],
      outputShape: [4, 4],
      inputTextureShape: [2, 1],
      outputTextureShape: [2, 2],
      expectedOutput: new Float32Array([1, 2, 5, 6, 3, 4, 7, 8, 1, 2, 5, 6, 3, 4, 7, 8]),
    },
    {
      elementCount: 8,
      axis: 1,
      inputShape: [2, 4],
      outputShape: [2, 8],
      inputTextureShape: [2, 1],
      outputTextureShape: [4, 2],
      expectedOutput: new Float32Array([
        1,
        2,
        5,
        6,
        1,
        2,
        5,
        6,
        3,
        4,
        7,
        8,
        3,
        4,
        7,
        8,
      ]),
    },
    {
      elementCount: 6,
      axis: 0,
      inputShape: [2, 3],
      outputShape: [4, 3],
      inputTextureShape: [2, 1],
      outputTextureShape: [2, 2],
      expectedOutput: new Float32Array([1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6]),
      rawInput: new Float32Array([1, 2, 4, 5, 3, 0, 6, 0])
    },
    {
      elementCount: 6,
      axis: 1,
      inputShape: [2, 3],
      outputShape: [2, 6],
      inputTextureShape: [2, 1],
      outputTextureShape: [2, 2],
      expectedOutput: new Float32Array([1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6]),
      rawInput: new Float32Array([1, 2, 4, 5, 3, 0, 6, 0])
    },

    // test 3d tensor
    {
      elementCount: 16,
      axis: 0,
      inputShape: [2, 2, 4],
      outputShape: [4, 2, 4],
      inputTextureShape: [2, 2],
      outputTextureShape: [2, 4],
      expectedOutput: new Float32Array([
        1, 2, 5, 6, 3, 4, 7, 8, 9, 10, 13, 14, 11, 12, 15, 16, 1, 2, 5, 6, 3, 4, 7, 8, 9, 10, 13, 14, 11, 12, 15, 16
      ])
    },
    {
      elementCount: 16,
      axis: 1,
      inputShape: [2, 2, 4],
      outputShape: [2, 4, 4],
      inputTextureShape: [2, 2],
      outputTextureShape: [4, 2],
      expectedOutput: new Float32Array([
        1, 2, 5, 6, 3, 4, 7, 8, 1, 2, 5, 6, 3, 4, 7, 8, 9, 10, 13, 14, 11, 12, 15, 16, 9, 10, 13, 14, 11, 12, 15, 16
      ])
    },
    {
      elementCount: 16,
      axis: 2,
      inputShape: [2, 2, 4],
      outputShape: [2, 2, 8],
      inputTextureShape: [2, 2],
      outputTextureShape: [4, 4],
      expectedOutput: new Float32Array([
        1, 2, 5, 6, 1, 2, 5, 6, 3, 4, 7, 8, 3, 4, 7, 8, 9, 10, 13, 14, 9, 10, 13, 14, 11, 12, 15, 16, 11, 12, 15, 16
      ])
    },

    // test 4d tensor
    {
      elementCount: 32,
      axis: 0,
      inputShape: [2, 2, 2, 4],
      outputShape: [4, 2, 2, 4],
      inputTextureShape: [2, 4],
      outputTextureShape: [2, 8],
      expectedOutput: new Float32Array([
        1,  2,  5,  6,  3,  4,  7,  8,  9,  10, 13, 14, 11, 12, 15, 16, 17, 18, 21, 22, 19, 20,
        23, 24, 25, 26, 29, 30, 27, 28, 31, 32, 1,  2,  5,  6,  3,  4,  7,  8,  9,  10, 13, 14,
        11, 12, 15, 16, 17, 18, 21, 22, 19, 20, 23, 24, 25, 26, 29, 30, 27, 28, 31, 32
      ])
    },
    {
      elementCount: 32,
      axis: 1,
      inputShape: [2, 2, 2, 4],
      outputShape: [2, 4, 2, 4],
      inputTextureShape: [2, 4],
      outputTextureShape: [8, 4],
      expectedOutput: new Float32Array([
        1,  2,  5,  6,  3,  4,  7,  8,  9,  10, 13, 14, 11, 12, 15, 16, 1,  2,  5,  6,  3,  4,
        7,  8,  9,  10, 13, 14, 11, 12, 15, 16, 17, 18, 21, 22, 19, 20, 23, 24, 25, 26, 29, 30,
        27, 28, 31, 32, 17, 18, 21, 22, 19, 20, 23, 24, 25, 26, 29, 30, 27, 28, 31, 32
      ])
    },

    {
      elementCount: 32,
      axis: 2,
      inputShape: [2, 2, 2, 4],
      outputShape: [2, 2, 4, 4],
      inputTextureShape: [2, 4],
      outputTextureShape: [8, 4],
      expectedOutput: new Float32Array([
        1,  2,  5,  6,  3,  4,  7,  8,  1,  2,  5,  6,  3,  4,  7,  8,  9,  10, 13, 14, 11, 12,
        15, 16, 9,  10, 13, 14, 11, 12, 15, 16, 17, 18, 21, 22, 19, 20, 23, 24, 17, 18, 21, 22,
        19, 20, 23, 24, 25, 26, 29, 30, 27, 28, 31, 32, 25, 26, 29, 30, 27, 28, 31, 32
      ])
    },
    {
      elementCount: 32,
      axis: 3,
      inputShape: [2, 2, 2, 4],
      outputShape: [2, 2, 4, 4],
      inputTextureShape: [2, 4],
      outputTextureShape: [8, 4],
      expectedOutput: new Float32Array([
        1,  2,  5,  6,  1,  2,  5,  6,  3,  4,  7,  8,  3,  4,  7,  8,  9,  10, 13, 14, 9,  10,
        13, 14, 11, 12, 15, 16, 11, 12, 15, 16, 17, 18, 21, 22, 17, 18, 21, 22, 19, 20, 23, 24,
        19, 20, 23, 24, 25, 26, 29, 30, 25, 26, 29, 30, 27, 28, 31, 32, 27, 28, 31, 32
      ])
    },
  ];
}

let backend: Backend|undefined;
let sessionhandler: SessionHandler|undefined;
let inferenceHandler: InferenceHandler|undefined;

describe('#UnitTest# - packed concat - Tensor concat', () => {
  before('Initialize Context', async () => {
    const profiler = Profiler.create();
    backend = await resolveBackend('webgl');
    // Explicitly set to true to trigger packed version
    (backend as WebGLBackend).pack = true;
    sessionhandler = backend.createSessionHandler({profiler});
    inferenceHandler = sessionhandler.createInferenceHandler();
  });

  // Set it back to false, apparently this state is sticky throughout all the tests running in same browser session..
  after('Resetting Context', () => {
    (backend as WebGLBackend).pack = false;
  });

  const testDataSet = getTestData();
  for (let k = 0; k < testDataSet.length; ++k) {
    const testData = testDataSet[k];
    describe(`Test concat ${JSON.stringify(testData)}`, () => {});
    it('Test packed concat kernel ', () => {
      const webglInferenceHandler = inferenceHandler as WebGLInferenceHandler;

      // TODO support WebGl 1.0
      if (webglInferenceHandler.session.textureManager.glContext.version === 1) {
        console.log('Running packed concat with webgl1 is not supported. Skipping.');
        return;
      }

      const op = new WebGLConcat();
      const attributes = new Attribute(undefined);
      const axis = testData.axis;
      attributes.set('axis', 'int', axis);

      op.initialize(attributes);
      const elementCount = testData.elementCount;
      const inputTensorShape = testData.inputShape;
      const inputTextureShape = testData.inputTextureShape;

      // create input data and tensor. The input data will be used to verify if the output tensor contains the
      // same value but possibly different order depending on our packing algorithm.
      const inputData = createAscendingArray(elementCount);
      const inputTensorA = new Tensor(inputTensorShape, 'float32', undefined, undefined, inputData);
      const inputTensorB = new Tensor(inputTensorShape, 'float32', undefined, undefined, inputData);

      // manually creat packed texture from inputTensor, and insert in cache
      const gl = webglInferenceHandler.session.textureManager.glContext.gl;
      webglInferenceHandler.session.textureManager.glContext.checkError();
      const webglTextureA = createTextureFromArray(
          webglInferenceHandler.session.textureManager.glContext, testData.rawInput ? testData.rawInput : inputData,
          gl.RGBA, inputTextureShape[0], inputTextureShape[1]);
      const webglTextureB = createTextureFromArray(
          webglInferenceHandler.session.textureManager.glContext, testData.rawInput ? testData.rawInput : inputData,
          gl.RGBA, inputTextureShape[0], inputTextureShape[1]);

      webglInferenceHandler.session.textureManager.glContext.checkError();
      const packedShape = inputTextureShape;
      const textureDataA = {
        width: inputTextureShape[0],
        height: inputTextureShape[1],
        channels: 4 as const,
        isPacked: true,
        shape: packedShape,
        strides: ShapeUtil.computeStrides(packedShape),
        unpackedShape: inputTensorShape,
        tensor: inputTensorA,
        texture: webglTextureA!
      };
      const textureDataB = {
        width: inputTextureShape[0],
        height: inputTextureShape[1],
        channels: 4 as const,
        isPacked: true,
        shape: packedShape,
        strides: ShapeUtil.computeStrides(packedShape),
        unpackedShape: inputTensorShape,
        tensor: inputTensorB,
        texture: webglTextureB!
      };

      webglInferenceHandler.setTextureData(inputTensorA.dataId, textureDataA, true);
      webglInferenceHandler.setTextureData(inputTensorB.dataId, textureDataB, true);

      // compile shader code
      const programInfo =
          op.createProgramInfo(inferenceHandler! as WebGLInferenceHandler, [inputTensorA, inputTensorB]);

      const artifact = webglInferenceHandler.session.programManager.build(programInfo, 'WebGLConcat');
      webglInferenceHandler.session.programManager.setArtifact(op, artifact);

      // run kernal and get output
      const runData = op.createRunData(webglInferenceHandler, artifact.programInfo, [inputTensorA, inputTensorB]);
      webglInferenceHandler.session.programManager.run(artifact, runData);
      const result = runData.outputTextureData.tensor.data;

      webglInferenceHandler.session.textureManager.glContext.checkError();
      // verify result.
      const expectedOutput = testData.expectedOutput;
      expect(result).to.not.equal(null);

      expect(result).to.have.lengthOf(elementCount * 2);

      expect(result).to.deep.equal(expectedOutput);
    });
  }
});
