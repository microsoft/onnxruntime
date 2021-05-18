// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {expect} from 'chai';

import {Backend, InferenceHandler, resolveBackend, SessionHandler} from '../../../../lib/onnxjs/backend';
import {WebGLBackend} from '../../../../lib/onnxjs/backends/backend-webgl';
import {WebGLInferenceHandler} from '../../../../lib/onnxjs/backends/webgl/inference-handler';
import {WebGLReshapePacked} from '../../../../lib/onnxjs/backends/webgl/ops/reshape-packed';
import {Profiler} from '../../../../lib/onnxjs/instrument';
import {Tensor} from '../../../../lib/onnxjs/tensor';

import {createAscendingArray} from './test-utils';

interface TestData {
  elementCount: number;
  inputShape: number[];
  outputShape: number[];
}
function getTestData(): TestData[] {
  return [
    // test 2D tensor
    {
      elementCount: 16,
      inputShape: [4, 4],
      outputShape: [2, 8],
    },
    {
      elementCount: 16,
      inputShape: [4, 4],
      outputShape: [1, 16],
    },
    {
      elementCount: 8,
      inputShape: [2, 4],
      outputShape: [4, 2],
    },
    {
      elementCount: 8,
      inputShape: [2, 4],
      outputShape: [1, 8],
    },
    {
      elementCount: 6,
      inputShape: [2, 3],
      outputShape: [1, 6],
    },
    {
      elementCount: 6,
      inputShape: [2, 3],
      outputShape: [3, 2],
    },

    // test 3d tensor
    {
      elementCount: 16,
      inputShape: [2, 2, 4],
      outputShape: [4, 2, 2],
    },
    {
      elementCount: 16,
      inputShape: [2, 2, 4],
      outputShape: [2, 4, 2],
    },
    {
      elementCount: 16,
      inputShape: [2, 2, 4],
      outputShape: [1, 1, 2, 8],
    },

    // test 4d tensor
    {
      elementCount: 32,
      inputShape: [2, 2, 2, 4],
      outputShape: [4, 2, 2, 2],
    },
    {
      elementCount: 32,
      inputShape: [2, 2, 2, 4],
      outputShape: [2, 4, 2, 2],
    },

    {
      elementCount: 32,
      inputShape: [2, 2, 2, 4],
      outputShape: [2, 2, 4, 2],
    },
    {
      elementCount: 32,
      inputShape: [2, 2, 2, 4],
      outputShape: [2, 1, 4, 4],
    },
    {
      elementCount: 18432,
      inputShape: [512, 36],
      outputShape: [512, 36, 1, 1],
    },
  ];
}

let backend: Backend|undefined;
let sessionhandler: SessionHandler|undefined;
let inferenceHandler: InferenceHandler|undefined;

describe('#UnitTest# - reshape - packed', () => {
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
    describe(`Test reshape ${JSON.stringify(testData)}`, () => {});
    it(`Test packed reshape kernel ${JSON.stringify(testData.outputShape)}`, () => {
      const webglInferenceHandler = inferenceHandler as WebGLInferenceHandler;

      // TODO support WebGl 1.0
      if (webglInferenceHandler.session.textureManager.glContext.version === 1) {
        console.log('Running packed concat with webgl1 is not supported. Skipping.');
        return;
      }

      const op = new WebGLReshapePacked();

      const elementCount = testData.elementCount;
      const inputTensorShape = testData.inputShape;
      const outputTensorShape = testData.outputShape;

      // create input data and tensor.
      const inputData = createAscendingArray(elementCount);
      const inputTensorA = new Tensor(inputTensorShape, 'float32', undefined, undefined, inputData);

      // create shape data tensor
      const inputTensorB =
          new Tensor([outputTensorShape.length], 'int32', undefined, undefined, new Int32Array(outputTensorShape));

      // compile shader code
      const programInfo =
          op.createProgramInfo(inferenceHandler! as WebGLInferenceHandler, [inputTensorA, inputTensorB]);

      const artifact = webglInferenceHandler.session.programManager.build(programInfo);
      webglInferenceHandler.session.programManager.setArtifact(op, artifact);

      // run kernal and get output
      const resultTensor = webglInferenceHandler.run(op, [inputTensorA, inputTensorB]);
      const result = resultTensor[0].data;

      webglInferenceHandler.session.textureManager.glContext.checkError();
      // verify result.
      expect(result).to.not.equal(null);

      expect(result).to.have.lengthOf(elementCount);

      expect(result).to.deep.equal(inputData);
    });
  }
});
