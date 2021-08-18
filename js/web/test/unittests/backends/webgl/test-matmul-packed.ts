// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {expect} from 'chai';
import {env} from 'onnxruntime-common';

import {Backend, InferenceHandler, resolveBackend, SessionHandler} from '../../../../lib/onnxjs/backend';
import {WebGLInferenceHandler} from '../../../../lib/onnxjs/backends/webgl/inference-handler';
import {createPackedMatmulProgramInfoLoader} from '../../../../lib/onnxjs/backends/webgl/ops/matmul-pack';
import {Profiler} from '../../../../lib/onnxjs/instrument';
import {Tensor} from '../../../../lib/onnxjs/tensor';

import {createAscendingArray} from './test-utils';

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
      rawInputA: new Float32Array([1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6]),
      rawInputB: new Float32Array([1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6]),
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
      rawInputA: new Float32Array([1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6]),
    },
    {
      elementCountA: 12,
      elementCountB: 6,
      inputShapeA: [1, 2, 2, 3],
      inputShapeB: [1, 1, 1, 3, 2],
      outputShape: [1, 1, 2, 2, 2],
      inputTextureShapeA: [2, 2],
      inputTextureShapeB: [1, 2],
      outputTextureShape: [2, 1],
      expectedOutput: new Float32Array([22, 28, 49, 64, 22, 28, 49, 64]),
      rawInputA: new Float32Array([1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6]),
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
    it(`Test packed matmul kernel [${testData.inputShapeA}]x[${testData.inputShapeB}]`, () => {
      const webglInferenceHandler = inferenceHandler as WebGLInferenceHandler;

      if (!env.webgl.pack) {
        console.log('Skipping in unpacked texture mode.');
        return;
      }

      const elementCountA = testData.elementCountA;
      const elementCountB = testData.elementCountB;

      const inputTensorShapeA = testData.inputShapeA;
      const inputTensorShapeB = testData.inputShapeB;

      // create input data and tensor. The input data will be used to verify if the output tensor contains the
      // same value but possibly different order depending on our packing algorithm.
      const inputDataA = testData.rawInputA ?? createAscendingArray(elementCountA);
      const inputDataB = testData.rawInputB ?? createAscendingArray(elementCountB);
      const inputTensorA = new Tensor(inputTensorShapeA, 'float32', undefined, undefined, inputDataA);
      const inputTensorB = new Tensor(inputTensorShapeB, 'float32', undefined, undefined, inputDataB);
      const biasTensor = testData.biasValue ?
          new Tensor([1], 'float32', undefined, undefined, new Float32Array([testData.biasValue])) :
          undefined;
      const inputs = biasTensor ? [inputTensorA, inputTensorB, biasTensor] : [inputTensorA, inputTensorB];

      const output = webglInferenceHandler.run(
          createPackedMatmulProgramInfoLoader(webglInferenceHandler, inputs, {activation: '', activationCacheKey: ''}),
          inputs);
      const result = output.data;

      webglInferenceHandler.session.textureManager.glContext.checkError();
      // verify result.
      const expectedOutput = testData.expectedOutput;
      expect(result).to.not.equal(null);
      let batchMultiplierA = 1;
      let batchMultiplierB = 1;

      if (testData.inputShapeA.length > 2) {
        for (let i = 0; i < testData.inputShapeA.length - 2; i++) {
          batchMultiplierA *= testData.inputShapeA[i];
        }
      }
      if (testData.inputShapeB.length > 2) {
        for (let i = 0; i < testData.inputShapeB.length - 2; i++) {
          batchMultiplierB *= testData.inputShapeB[i];
        }
      }
      const batchMultiplier = Math.max(batchMultiplierA, batchMultiplierB);
      expect(result).to.have.lengthOf(
          batchMultiplier * testData.inputShapeA[testData.inputShapeA.length - 2] *
          testData.inputShapeB[testData.inputShapeB.length - 1]);
      expect(result).to.deep.equal(expectedOutput);
    });
  }
});
