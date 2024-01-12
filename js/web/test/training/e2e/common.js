// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

'use strict';

const DATA_FOLDER = 'data/';
const TRAININGDATA_TRAIN_MODEL = DATA_FOLDER + 'training_model.onnx';
const TRAININGDATA_OPTIMIZER_MODEL = DATA_FOLDER + 'adamw.onnx';
const TRAININGDATA_EVAL_MODEL = DATA_FOLDER + 'eval_model.onnx';
const TRAININGDATA_CKPT = DATA_FOLDER + 'checkpoint.ckpt';

const trainingSessionAllOptions = {
  checkpointState: TRAININGDATA_CKPT,
  trainModel: TRAININGDATA_TRAIN_MODEL,
  evalModel: TRAININGDATA_EVAL_MODEL,
  optimizerModel: TRAININGDATA_OPTIMIZER_MODEL
}

const trainingSessionMinOptions = {
  checkpointState: TRAININGDATA_CKPT,
  trainModel: TRAININGDATA_TRAIN_MODEL,
}

// ASSERT METHODS

function assert(cond) {
  if (!cond) throw new Error();
}

function assertStrictEquals(actual, expected) {
  if (actual !== expected) {
    let strRep = actual;
    if (typeof actual === 'object') {
      strRep = JSON.stringify(actual);
    }
    throw new Error(`expected: ${expected}; got: ${strRep}`);
  }
}

function assertTwoListsUnequal(list1, list2) {
  if (list1.length !== list2.length) {
    return;
  }
  for (let i = 0; i < list1.length; i++) {
    if (list1[i] !== list2[i]) {
      return;
    }
  }
  throw new Error(`expected ${list1} and ${list2} to be unequal; got two equal lists`);
}

// HELPER METHODS FOR TESTS

function generateGaussianRandom(mean=0, scale=1) {
  const u = 1 - Math.random();
  const v = Math.random();
  const z = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  return z * scale + mean;
}

function generateGaussianFloatArray(length) {
  const array = new Float32Array(length);

  for (let i = 0; i < length; i++) {
    array[i] = generateGaussianRandom();
  }

  return array;
}

/**
 * creates the TrainingSession and verifies that the input and output names of the training model loaded into the
 * training session are correct.
 * @param {} ort
 * @param {*} createOptions
 * @param {*} options
 * @returns
 */
async function createTrainingSessionAndCheckTrainingModel(ort, createOptions, options) {
  const trainingSession = await ort.TrainingSession.create(createOptions, options);

  assertStrictEquals(trainingSession.trainingInputNames[0], 'input-0');
  assertStrictEquals(trainingSession.trainingInputNames[1], 'labels');
  assertStrictEquals(trainingSession.trainingInputNames.length, 2);
  assertStrictEquals(trainingSession.trainingOutputNames[0], 'onnx::loss::21273');
  assertStrictEquals(trainingSession.trainingOutputNames.length, 1);
  return trainingSession;
}

/**
 * verifies that the eval input and output names associated with the eval model loaded into the given training session
 * are correct.
 */
function checkEvalModel(trainingSession) {
  assertStrictEquals(trainingSession.evalInputNames[0], 'input-0');
  assertStrictEquals(trainingSession.evalInputNames[1], 'labels');
  assertStrictEquals(trainingSession.evalInputNames.length, 2);
  assertStrictEquals(trainingSession.evalOutputNames[0], 'onnx::loss::21273');
  assertStrictEquals(trainingSession.evalOutputNames.length, 1);
}

/**
 * Checks that accessing trainingSession.evalInputNames or trainingSession.evalOutputNames will throw an error if
 * accessed
 * @param {} trainingSession
 */
function checkNoEvalModel(trainingSession) {
  try {
    assertStrictEquals(trainingSession.evalInputNames, "should have thrown an error upon accessing");
  } catch (error) {
    assertStrictEquals(error.message, 'This training session has no evalModel loaded.');
  }
  try {
    assertStrictEquals(trainingSession.evalOutputNames, "should have thrown an error upon accessing");
  } catch (error) {
    assertStrictEquals(error.message, 'This training session has no evalModel loaded.');
  }
}

/**
 * runs the train step with the given inputs and checks that the tensor returned is of type float32 and has a length
 * of 1 for the loss.
 * @param {} trainingSession
 * @param {*} feeds
 * @returns
 */
var runTrainStepAndCheck = async function(trainingSession, feeds) {
  const results =  await trainingSession.runTrainStep(feeds);
  assertStrictEquals(Object.keys(results).length, 1);
  assertStrictEquals(results['onnx::loss::21273'].data.length, 1);
  assertStrictEquals(results['onnx::loss::21273'].type, 'float32');
  return results;
};

var loadParametersBufferAndCheck = async function(trainingSession, paramsLength, constant, paramsBefore) {
  // make a float32 array that is filled with the constant
  const newParams = new Float32Array(paramsLength);
  for (let i = 0; i < paramsLength; i++) {
    newParams[i] = constant;
  }

  const newParamsUint8 = new Uint8Array(newParams.buffer, newParams.byteOffset, newParams.byteLength);

  await trainingSession.loadParametersBuffer(newParamsUint8);
  const paramsAfterLoad = await trainingSession.getContiguousParameters();

  // check that the parameters have changed
  assertTwoListsUnequal(paramsAfterLoad.data, paramsBefore.data);
  assertStrictEquals(paramsAfterLoad.dims[0], paramsLength);

  // check that the parameters have changed to what they should be
  for (let i = 0; i < paramsLength; i++) {
    // round to the same number of digits (4 decimal places)
    assertStrictEquals(paramsAfterLoad.data[i].toFixed(4), constant.toFixed(4));
  }

  return paramsAfterLoad;
}

// TESTS

var testInferenceFunction = async function(ort, options) {
  const session = await ort.InferenceSession.create('data/model.onnx', options || {});

  const dataA = Float32Array.from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
  const dataB = Float32Array.from([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]);

  const fetches =
      await session.run({a: new ort.Tensor('float32', dataA, [3, 4]), b: new ort.Tensor('float32', dataB, [4, 3])});

  const c = fetches.c;

  assert(c instanceof ort.Tensor);
  assert(c.dims.length === 2 && c.dims[0] === 3 && c.dims[1] === 3);
  assert(c.data[0] === 700);
  assert(c.data[1] === 800);
  assert(c.data[2] === 900);
  assert(c.data[3] === 1580);
  assert(c.data[4] === 1840);
  assert(c.data[5] === 2100);
  assert(c.data[6] === 2460);
  assert(c.data[7] === 2880);
  assert(c.data[8] === 3300);
};

var testTrainingFunctionMin = async function(ort, options) {
  const trainingSession = await createTrainingSessionAndCheckTrainingModel(ort, trainingSessionMinOptions, options);
  checkNoEvalModel(trainingSession);
  const input0 = new ort.Tensor('float32', generateGaussianFloatArray(2 * 784), [2, 784]);
  const labels = new ort.Tensor('int32', [2, 1], [2]);
  const feeds = {"input-0": input0, "labels": labels};

  // check getParametersSize
  const paramsSize = await trainingSession.getParametersSize();
  assertStrictEquals(paramsSize, 397510);

  // check getContiguousParameters
  const originalParams = await trainingSession.getContiguousParameters();
  assertStrictEquals(originalParams.dims.length, 1);
  assertStrictEquals(originalParams.dims[0], 397510);
  assertStrictEquals(originalParams.data[0], -0.025190064683556557);
  assertStrictEquals(originalParams.data[2000], -0.034044936299324036);

  await runTrainStepAndCheck(trainingSession, feeds);

  await loadParametersBufferAndCheck(trainingSession, 397510, -1.2, originalParams);
}

var testTrainingFunctionAll = async function(ort, options) {
  const trainingSession = await createTrainingSessionAndCheckTrainingModel(ort, trainingSessionAllOptions, options);
  checkEvalModel(trainingSession);

  const input0 = new ort.Tensor('float32', generateGaussianFloatArray(2 * 784), [2, 784]);
  const labels = new ort.Tensor('int32', [2, 1], [2]);
  let feeds = {"input-0": input0, "labels": labels};

  // check getParametersSize
  const paramsSize = await trainingSession.getParametersSize();
  assertStrictEquals(paramsSize, 397510);

  // check getContiguousParameters
  const originalParams = await trainingSession.getContiguousParameters();
  assertStrictEquals(originalParams.dims.length, 1);
  assertStrictEquals(originalParams.dims[0], 397510);
  assertStrictEquals(originalParams.data[0], -0.025190064683556557);
  assertStrictEquals(originalParams.data[2000], -0.034044936299324036);

  const results = await runTrainStepAndCheck(trainingSession, feeds);

  await trainingSession.runOptimizerStep(feeds);
  feeds = {"input-0": input0, "labels": labels};
  // check getContiguousParameters after optimizerStep -- that the parameters have been updated
  const optimizedParams = await trainingSession.getContiguousParameters();
  assertTwoListsUnequal(originalParams.data, optimizedParams.data);

  const results2 = await runTrainStepAndCheck(trainingSession, feeds);

  // check that loss decreased after optimizer step and training again
  assert(results2['onnx::loss::21273'].data < results['onnx::loss::21273'].data);

  await loadParametersBufferAndCheck(trainingSession, 397510, -1.2, optimizedParams);
}

if (typeof module === 'object') {
  module.exports = [testInferenceFunction, testTrainingFunctionMin, testTrainingFunctionAll, testTest];
}
