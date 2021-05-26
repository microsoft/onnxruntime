// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {expect} from 'chai';

import {Attribute} from '../../../../lib/onnxjs/attribute';
import {Backend, InferenceHandler, resolveBackend, SessionHandler} from '../../../../lib/onnxjs/backend';
import {WebGLBackend} from '../../../../lib/onnxjs/backends/backend-webgl';
import {WebGLInferenceHandler} from '../../../../lib/onnxjs/backends/webgl/inference-handler';
import {WebGLDepthToSpace} from '../../../../lib/onnxjs/backends/webgl/ops/depth-to-space';
import {Profiler} from '../../../../lib/onnxjs/instrument';
import {Tensor} from '../../../../lib/onnxjs/tensor';

import {createAscendingArray} from './test-utils';

interface TestData {
  elementCount: number;
  blocksize: number;
  inputShape: number[];
  outputShape: number[];
  inputTextureShape: number[];
  outputTextureShape: number[];
  expectedOutput: Float32Array;
  // If empty, the test will use auto-generated data.
  rawInput?: Float32Array;
  mode?: string;
}
function getTestData(): TestData[] {
  return [
    {
      elementCount: 8,
      blocksize: 2,
      inputShape: [1, 8, 1, 1],
      outputShape: [1, 2, 2, 2],
      inputTextureShape: [8, 1],
      outputTextureShape: [4, 2],
      rawInput: new Float32Array([0., 9., 18., 27., 36., 45., 54., 63.]),
      expectedOutput: new Float32Array([0., 18., 36., 54., 9., 27., 45., 63.]),
      mode: 'DCR',
    },
    {
      elementCount: 16,
      blocksize: 2,
      inputShape: [1, 8, 1, 2],
      outputShape: [1, 2, 4, 2],
      inputTextureShape: [1, 16],
      outputTextureShape: [8, 2],
      rawInput: new Float32Array([0., 1., 9., 10, 18., 19, 27., 28., 36., 37., 45., 46., 54., 55., 63., 64.]),
      expectedOutput: new Float32Array([0, 18, 1, 19, 36, 54, 37, 55, 9, 27, 10, 28, 45, 63, 46, 64]),
      mode: 'DCR',
    },

    {
      elementCount: 48,
      blocksize: 2,
      inputShape: [1, 8, 2, 3],
      outputShape: [1, 2, 4, 6],
      inputTextureShape: [16, 3],
      outputTextureShape: [8, 6],
      rawInput: new Float32Array([
        0.,  1.,  2.,  3.,  4.,  5.,  9.,  10., 11., 12., 13., 14., 18., 19., 20., 21.,
        22., 23., 27., 28., 29., 30., 31., 32., 36., 37., 38., 39., 40., 41., 45., 46.,
        47., 48., 49., 50., 54., 55., 56., 57., 58., 59., 63., 64., 65., 66., 67., 68.
      ]),
      expectedOutput: new Float32Array([
        0.,  18., 1.,  19., 2.,  20., 36., 54., 37., 55., 38., 56., 3.,  21., 4.,  22.,
        5.,  23., 39., 57., 40., 58., 41., 59., 9.,  27., 10., 28., 11., 29., 45., 63.,
        46., 64., 47., 65., 12., 30., 13., 31., 14., 32., 48., 66., 49., 67., 50., 68.
      ]),
      mode: 'DCR',
    },
    {
      elementCount: 8,
      blocksize: 2,
      inputShape: [1, 8, 1, 1],
      outputShape: [1, 2, 2, 2],
      inputTextureShape: [8, 1],
      outputTextureShape: [4, 2],
      rawInput: new Float32Array([0., 9., 18., 27., 36., 45., 54., 63.]),
      expectedOutput: new Float32Array([0, 9, 18, 27, 36, 45, 54, 63]),
      mode: 'CRD',
    },
    {
      elementCount: 16,
      blocksize: 2,
      inputShape: [1, 8, 1, 2],
      outputShape: [1, 2, 4, 2],
      inputTextureShape: [1, 16],
      outputTextureShape: [8, 2],
      rawInput: new Float32Array([0., 1., 9., 10, 18., 19, 27., 28., 36., 37., 45., 46., 54., 55., 63., 64.]),
      expectedOutput: new Float32Array([0, 9, 1, 10, 18, 27, 19, 28, 36, 45, 37, 46, 54, 63, 55, 64]),
      mode: 'CRD',
    },

    {
      elementCount: 48,
      blocksize: 2,
      inputShape: [1, 8, 2, 3],
      outputShape: [1, 2, 4, 6],
      inputTextureShape: [16, 3],
      outputTextureShape: [8, 6],
      rawInput: new Float32Array([
        0.,  1.,  2.,  3.,  4.,  5.,  9.,  10., 11., 12., 13., 14., 18., 19., 20., 21.,
        22., 23., 27., 28., 29., 30., 31., 32., 36., 37., 38., 39., 40., 41., 45., 46.,
        47., 48., 49., 50., 54., 55., 56., 57., 58., 59., 63., 64., 65., 66., 67., 68.
      ]),
      expectedOutput: new Float32Array([
        0.,  9.,  1.,  10., 2.,  11., 18., 27., 19., 28., 20., 29., 3.,  12., 4.,  13.,
        5.,  14., 21., 30., 22., 31., 23., 32., 36., 45., 37., 46., 38., 47., 54., 63.,
        55., 64., 56., 65., 39., 48., 40., 49., 41., 50., 57., 66., 58., 67., 59., 68.
      ]),
      mode: 'CRD',
    },
  ];
}

let backend: Backend|undefined;
let sessionhandler: SessionHandler|undefined;
let inferenceHandler: InferenceHandler|undefined;

describe('#UnitTest# - unpacked WebGLDepthToSpace - Tensor WebGLDepthToSpace', () => {
  before('Initialize Context', async () => {
    const profiler = Profiler.create();
    backend = await resolveBackend('webgl');
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
    it('Test depth to space ', () => {
      const webglInferenceHandler = inferenceHandler as WebGLInferenceHandler;

      // TODO support WebGl 1.0
      if (webglInferenceHandler.session.textureManager.glContext.version === 1) {
        console.log('Running depth to space with webgl1 is not supported. Skipping.');
        return;
      }

      const op = new WebGLDepthToSpace();
      const attributes = new Attribute(undefined);
      const blocksize = testData.blocksize;
      attributes.set('blocksize', 'int', blocksize);
      attributes.set('mode', 'string', testData.mode as string);

      op.initialize(attributes);
      const elementCount = testData.elementCount;
      const inputTensorShape = testData.inputShape;

      // create input data and tensor.
      const inputData = testData.rawInput ? testData.rawInput : createAscendingArray(elementCount);
      const inputTensorA = new Tensor(inputTensorShape, 'float32', undefined, undefined, inputData);

      // manually creat packed texture from inputTensor, and insert in cache
      webglInferenceHandler.session.textureManager.glContext.checkError();

      webglInferenceHandler.session.textureManager.glContext.checkError();

      const result = op.run(webglInferenceHandler, [inputTensorA]);

      webglInferenceHandler.session.textureManager.glContext.checkError();
      // verify result.
      const expectedOutput = testData.expectedOutput;
      expect(result[0].data).to.not.equal(null);

      expect(result[0].data).to.have.lengthOf(elementCount);
      expect(result[0].data).to.deep.equal(expectedOutput);
    });
  }
});
