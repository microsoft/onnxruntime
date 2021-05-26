// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {expect} from 'chai';
import {env} from 'onnxruntime-common';

import {Backend, InferenceHandler, resolveBackend, SessionHandler} from '../../../../lib/onnxjs/backend';
import {WebGLInferenceHandler} from '../../../../lib/onnxjs/backends/webgl/inference-handler';
import {WebGLMatMulPacked} from '../../../../lib/onnxjs/backends/webgl/ops/matmul-pack';
import {Profiler} from '../../../../lib/onnxjs/instrument';
import {Tensor} from '../../../../lib/onnxjs/tensor';
import {ShapeUtil} from '../../../../lib/onnxjs/util';

import {createAscendingArray, createTextureFromArray} from './test-utils';

interface TestData {
  elementCountA: number;
  elementCountB: number;
  inputShapeA: number[];
  inputShapeB: number[];
  outputShape: number[];
  inputTextureShapeA: number[];
  inputTextureShapeB: number[];
  outputTextureShape: number[];
  expectedOutput: Float32Array;
  // The value of bias matrix that will be broadcasted to the corresponding shape in matmul.
  // i.e. If biasValue = 1, then bias matrix is [1], when being added to 2x2 matmul result, it will be bcasted to
  // [1, 1]
  // [1, 1]
  biasValue?: number;
  // If empty, the test will use auto-generated data.
  rawInputA?: Float32Array;
  // If empty, the test will use auto-generated data.
  rawInputB?: Float32Array;
}
function getTestData(): TestData[] {
  return [
    // test 2D tensor
    {
      elementCountA: 4,
      elementCountB: 4,
      inputShapeA: [2, 2],
      inputShapeB: [2, 2],
      outputShape: [2, 2],
      inputTextureShapeA: [1, 1],
      inputTextureShapeB: [1, 1],
      outputTextureShape: [1, 1],
      expectedOutput: new Float32Array([7, 10, 15, 22]),
    },
    {
      elementCountA: 4,
      elementCountB: 4,
      inputShapeA: [2, 2],
      inputShapeB: [2, 2],
      outputShape: [2, 2],
      inputTextureShapeA: [1, 1],
      inputTextureShapeB: [1, 1],
      outputTextureShape: [1, 1],
      biasValue: 1,
      expectedOutput: new Float32Array([8, 11, 16, 23]),
    },
    {
      elementCountA: 6,
      elementCountB: 6,
      inputShapeA: [2, 3],
      inputShapeB: [3, 2],
      outputShape: [2, 2],
      inputTextureShapeA: [2, 1],
      inputTextureShapeB: [1, 2],
      outputTextureShape: [1, 1],
      expectedOutput: new Float32Array([22, 28, 49, 64]),
      rawInputA: new Float32Array([1, 2, 4, 5, 3, 0, 6, 0]),
      rawInputB: new Float32Array([1, 2, 3, 4, 5, 6, 0, 0]),
    },
    {
      elementCountA: 6,
      elementCountB: 6,
      inputShapeA: [2, 3],
      inputShapeB: [3, 2],
      outputShape: [2, 2],
      inputTextureShapeA: [2, 1],
      inputTextureShapeB: [1, 2],
      outputTextureShape: [1, 1],
      expectedOutput: new Float32Array([23, 29, 50, 65]),
      biasValue: 1,
      rawInputA: new Float32Array([1, 2, 4, 5, 3, 0, 6, 0]),
      rawInputB: new Float32Array([1, 2, 3, 4, 5, 6, 0, 0]),
    },
    {
      elementCountA: 16,
      elementCountB: 16,
      inputShapeA: [4, 4],
      inputShapeB: [4, 4],
      outputShape: [4, 4],
      inputTextureShapeA: [2, 2],
      inputTextureShapeB: [2, 2],
      outputTextureShape: [2, 2],
      rawInputA: new Float32Array([1, 2, 5, 6, 3, 4, 7, 8, 9, 10, 13, 14, 11, 12, 15, 16]),
      rawInputB: new Float32Array([1, 2, 5, 6, 3, 4, 7, 8, 9, 10, 13, 14, 11, 12, 15, 16]),
      biasValue: 2,
      expectedOutput: new Float32Array([92, 102, 112, 122, 204, 230, 256, 282, 316, 358, 400, 442, 428, 486, 544, 602]),
    },
    {
      elementCountA: 12,
      elementCountB: 12,
      inputShapeA: [2, 2, 3],
      inputShapeB: [2, 3, 2],
      outputShape: [2, 2, 2],
      inputTextureShapeA: [2, 2],
      inputTextureShapeB: [1, 4],
      outputTextureShape: [2, 1],
      expectedOutput: new Float32Array([23, 29, 50, 65, 23, 29, 50, 65]),
      biasValue: 1,
      rawInputA: new Float32Array([1, 2, 4, 5, 3, 0, 6, 0, 1, 2, 4, 5, 3, 0, 6, 0]),
      rawInputB: new Float32Array([1, 2, 3, 4, 5, 6, 0, 0, 1, 2, 3, 4, 5, 6, 0, 0]),
    },
    // test bcast
    {
      elementCountA: 12,
      elementCountB: 6,
      inputShapeA: [2, 2, 3],
      inputShapeB: [3, 2],
      outputShape: [2, 2, 2],
      inputTextureShapeA: [2, 2],
      inputTextureShapeB: [1, 2],
      outputTextureShape: [2, 1],
      expectedOutput: new Float32Array([23, 29, 50, 65, 23, 29, 50, 65]),
      biasValue: 1,
      rawInputA: new Float32Array([1, 2, 4, 5, 3, 0, 6, 0, 1, 2, 4, 5, 3, 0, 6, 0]),
      rawInputB: new Float32Array([1, 2, 3, 4, 5, 6, 0, 0]),
    },
  ];
}

let backend: Backend|undefined;
let sessionhandler: SessionHandler|undefined;
let inferenceHandler: InferenceHandler|undefined;

describe('#UnitTest# - packed matmul - Tensor matmul', () => {
  before('Initialize Context', async () => {
    const profiler = Profiler.create();
    backend = await resolveBackend('webgl');
    sessionhandler = backend.createSessionHandler({profiler});
    inferenceHandler = sessionhandler.createInferenceHandler();
  });

  const testDataSet = getTestData();
  for (let k = 0; k < testDataSet.length; ++k) {
    const testData = testDataSet[k];
    describe(`Test matmul ${JSON.stringify(testData)}`, () => {});
    it('Test packed matmul kernel ', () => {
      const webglInferenceHandler = inferenceHandler as WebGLInferenceHandler;

      // TODO support WebGl 1.0
      if (webglInferenceHandler.session.textureManager.glContext.version === 1) {
        console.log('Running packed matmul with webgl1 is not supported. Skipping.');
        return;
      }

      if (env.webgl.pack === false) {
        console.log('Skipping in unpacked texture mode.');
        return;
      }

      const op = new WebGLMatMulPacked();

      const elementCountA = testData.elementCountA;
      const elementCountB = testData.elementCountB;

      const inputTensorShapeA = testData.inputShapeA;
      const inputTextureShapeA = testData.inputTextureShapeA;

      const inputTensorShapeB = testData.inputShapeB;
      const inputTextureShapeB = testData.inputTextureShapeB;

      // create input data and tensor. The input data will be used to verify if the output tensor contains the
      // same value but possibly different order depending on our packing algorithm.
      const inputDataA = createAscendingArray(elementCountA);
      const inputDataB = createAscendingArray(elementCountB);
      const inputTensorA = new Tensor(inputTensorShapeA, 'float32', undefined, undefined, inputDataA);
      const inputTensorB = new Tensor(inputTensorShapeB, 'float32', undefined, undefined, inputDataB);

      // manually creat packed texture from inputTensor, and insert in cache
      const gl = webglInferenceHandler.session.textureManager.glContext.gl;

      webglInferenceHandler.session.textureManager.glContext.checkError();
      const webglTextureA = createTextureFromArray(
          webglInferenceHandler.session.textureManager.glContext, testData.rawInputA ? testData.rawInputA : inputDataA,
          gl.RGBA, inputTextureShapeA[0], inputTextureShapeA[1]);
      const webglTextureB = createTextureFromArray(
          webglInferenceHandler.session.textureManager.glContext, testData.rawInputB ? testData.rawInputB : inputDataB,
          gl.RGBA, inputTextureShapeB[0], inputTextureShapeB[1]);

      webglInferenceHandler.session.textureManager.glContext.checkError();
      const packedShapeA = inputTextureShapeA;
      const textureDataA = {
        width: inputTextureShapeA[0],
        height: inputTextureShapeA[1],
        channels: 4 as const,
        isPacked: true,
        shape: packedShapeA,
        strides: ShapeUtil.computeStrides(packedShapeA),
        unpackedShape: inputTensorShapeA,
        tensor: inputTensorA,
        texture: webglTextureA!
      };

      const packedShapeB = inputTextureShapeB;
      const textureDataB = {
        width: inputTextureShapeB[0],
        height: inputTextureShapeB[1],
        channels: 4 as const,
        isPacked: true,
        shape: packedShapeB,
        strides: ShapeUtil.computeStrides(packedShapeB),
        unpackedShape: inputTensorShapeB,
        tensor: inputTensorB,
        texture: webglTextureB!
      };

      webglInferenceHandler.setTextureData(inputTensorA.dataId, textureDataA, true);
      webglInferenceHandler.setTextureData(inputTensorB.dataId, textureDataB, true);

      const inputList = testData.biasValue ?
          [
            inputTensorA, inputTensorB,
            new Tensor([1], 'float32', undefined, undefined, new Float32Array([testData.biasValue]))
          ] :
          [inputTensorA, inputTensorB];

      // compile shader code
      const programInfo = op.createProgramInfo(inferenceHandler! as WebGLInferenceHandler, inputList);

      const artifact = webglInferenceHandler.session.programManager.build(programInfo);
      webglInferenceHandler.session.programManager.setArtifact(op, artifact);

      // run kernal and get output
      const runData = op.createRunData(webglInferenceHandler, artifact.programInfo, inputList);
      webglInferenceHandler.session.programManager.run(artifact, runData);
      const result = runData.outputTextureData.tensor.data;

      webglInferenceHandler.session.textureManager.glContext.checkError();
      // verify result.
      const expectedOutput = testData.expectedOutput;
      expect(result).to.not.equal(null);
      let batchMultiplier = 1;
      if (testData.inputShapeA.length > 2) {
        batchMultiplier = testData.inputShapeA[0];
      }
      if (testData.inputShapeB.length > 2) {
        batchMultiplier = Math.max(batchMultiplier, testData.inputShapeB[0]);
      }

      expect(result).to.have.lengthOf(
          batchMultiplier * testData.inputShapeA[testData.inputShapeA.length - 2] *
          testData.inputShapeB[testData.inputShapeB.length - 1]);
      expect(result).to.deep.equal(expectedOutput);
    });
  }
});
