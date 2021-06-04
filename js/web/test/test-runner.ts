// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {expect} from 'chai';
import {readFile} from 'fs';
import {onnx as onnxProto} from 'onnx-proto';
import * as ort from 'onnxruntime-common';
import {extname} from 'path';
import {inspect, promisify} from 'util';

import {Attribute} from '../lib/onnxjs/attribute';
import {InferenceHandler, resolveBackend, SessionHandler} from '../lib/onnxjs/backend';
import {createWebGLContext} from '../lib/onnxjs/backends/webgl/webgl-context-factory';
import {Logger, Profiler} from '../lib/onnxjs/instrument';
import {Operator} from '../lib/onnxjs/operators';
import {Tensor} from '../lib/onnxjs/tensor';

import {base64toBuffer, createMockGraph} from './test-shared';
import {Test} from './test-types';

// the threshold that used to compare 2 float numbers. See above for TensorResultValidator.floatEqual().
const CPU_THRESHOLD_ABSOLUTE_ERROR = 1.0e-4;
const CPU_THRESHOLD_RELATIVE_ERROR = 1.000001;
const WEBGL_THRESHOLD_ABSOLUTE_ERROR = 1.0e-3;
const WEBGL_THRESHOLD_RELATIVE_ERROR = 1.00001;
const WEBGL_HALF_FLOAT_THRESHOLD_ABSOLUTE_ERROR = 0.1;
const WEBGL_HALF_FLOAT_THRESHOLD_RELATIVE_ERROR = 1.02;
const WASM_THRESHOLD_ABSOLUTE_ERROR = 1.0e-4;
const WASM_THRESHOLD_RELATIVE_ERROR = 1.000001;
const ONNXRUNTIME_THRESHOLD_ABSOLUTE_ERROR = 1.0e-3;
const ONNXRUNTIME_THRESHOLD_RELATIVE_ERROR = 1.00001;

/**
 * returns a number to represent the current timestamp in a resolution as high as possible.
 */
const now = (typeof performance !== 'undefined' && performance.now) ? () => performance.now() : Date.now;

function toInternalTensor(tensor: ort.Tensor): Tensor {
  return new Tensor(
      tensor.dims, tensor.type as Tensor.DataType, undefined, undefined, tensor.data as Tensor.NumberType);
}
function fromInternalTensor(tensor: Tensor): ort.Tensor {
  return new ort.Tensor(tensor.type, tensor.data as ort.Tensor.DataType, tensor.dims);
}

async function loadFile(uri: string): Promise<Uint8Array|ArrayBuffer> {
  if (typeof fetch === 'undefined') {
    // node
    return promisify(readFile)(uri);
  } else {
    // browser
    const response = await fetch(uri);
    return response.arrayBuffer();
  }
}

async function loadTensorProto(uriOrData: string|Uint8Array): Promise<Test.NamedTensor> {
  const buf = (typeof uriOrData === 'string') ? await loadFile(uriOrData) : uriOrData;
  const tensorProto = onnxProto.TensorProto.decode(Buffer.from(buf));
  const tensor = Tensor.fromProto(tensorProto);
  // add property 'name' to the tensor object.
  const namedTensor = fromInternalTensor(tensor) as unknown as Test.NamedTensor;
  namedTensor.name = tensorProto.name;
  return namedTensor;
}

async function loadMlProto(_uriOrData: string|Uint8Array): Promise<Test.NamedTensor> {
  return Promise.reject('not supported');
}

async function loadTensors(
    modelMetaData: {inputNames: readonly string[]; outputNames: readonly string[]}, testCase: Test.ModelTestCase,
    fileCache?: FileCacheBuffer) {
  const inputs: Test.NamedTensor[] = [];
  const outputs: Test.NamedTensor[] = [];
  let dataFileType: 'none'|'pb'|'npy' = 'none';

  for (const dataFile of testCase.dataFiles) {
    const ext = extname(dataFile);
    if (ext.toLowerCase() === '.pb' || ext.toLowerCase() === '.tpb') {
      if (dataFileType === 'none') {
        dataFileType = 'pb';
      }
      if (dataFileType !== 'pb') {
        throw new Error(`cannot load data from test case "${testCase.name}", multiple types of files detected`);
      }

      const uriOrData = fileCache && fileCache[dataFile] ? fileCache[dataFile] : dataFile;
      const t = ext.toLowerCase() === '.pb' ? await loadTensorProto(uriOrData) :  // onnx.TensorProto
          await loadMlProto(uriOrData);                                           // (TBD)

      const dataFileBasename = dataFile.split(/[/\\]/).pop()!;

      if (dataFileBasename.indexOf('input') !== -1) {
        inputs.push(t);
      } else if (dataFileBasename.indexOf('output') !== -1) {
        outputs.push(t);
      }
    } else {
      throw new Error(`${ext} file is not supported now`);
    }
  }

  // if model has single input/output, and tensor name is empty, we assign model's input/output names to it.
  if (modelMetaData.inputNames.length === 1 && inputs.length === 1 && !inputs[0].name) {
    inputs[0].name = modelMetaData.inputNames[0];
  }
  if (modelMetaData.outputNames.length === 1 && outputs.length === 1 && !outputs[0].name) {
    outputs[0].name = modelMetaData.outputNames[0];
  }

  testCase.inputs = inputs;
  testCase.outputs = outputs;
}

async function initializeSession(
    modelFilePath: string, backendHint: string, profile: boolean,
    fileCache?: FileCacheBuffer): Promise<ort.InferenceSession> {
  const preloadModelData: Uint8Array|undefined =
      fileCache && fileCache[modelFilePath] ? fileCache[modelFilePath] : undefined;
  Logger.verbose(
      'TestRunner',
      `Start to load model from file: ${modelFilePath}${
          preloadModelData ? ` [preloaded(${preloadModelData.byteLength})]` : ''}`);

  const profilerConfig = profile ? {maxNumberEvents: 65536} : undefined;
  const sessionConfig = {executionProviders: [backendHint], profiler: profilerConfig};
  let session: ort.InferenceSession;

  try {
    if (preloadModelData) {
      session = await ort.InferenceSession.create(preloadModelData, sessionConfig);
    } else {
      session = await ort.InferenceSession.create(modelFilePath, sessionConfig);
    }
  } catch (e) {
    Logger.error('TestRunner', `Failed to load model from file: ${modelFilePath}. Error: ${inspect(e)}`);
    throw e;
  }

  if (profile) {
    session.startProfiling();
  }

  Logger.verbose('TestRunner', `Finished loading model from file: ${modelFilePath}`);

  return session;
}

type FileCacheBuffer = {
  [filePath: string]: Uint8Array;
};
/**
 * a ModelTestContext object contains all states in a ModelTest
 */
export class ModelTestContext {
  private constructor(
      readonly session: ort.InferenceSession,
      readonly backend: string,
      readonly perfData: ModelTestContext.ModelTestPerfData,
      private readonly profile: boolean,
  ) {}

  /**
   * dump the current performance data
   */
  private logPerfData() {
    const data = this.perfData;
    Logger.verbose('TestRunner.Perf', '***Perf Data Start');
    Logger.verbose('TestRunner.Perf', ` * Init          : ${data.init}`);
    Logger.verbose('TestRunner.Perf', ` * Running times : ${data.count}`);
    Logger.verbose('TestRunner.Perf', ` * FirstRun      : ${data.firstRun.toFixed(2)}`);
    const runs = data.runs;
    if (runs.length > 0) {
      Logger.verbose('TestRunner.Perf', ` * Runs          : ${runs.map(r => r.toFixed(2)).join(', ')}`);

      if (runs.length > 1) {
        const avg = runs.reduce((prev, current) => prev + current) / runs.length;
        Logger.verbose('TestRunner.Perf', ` * Runs Avg      : ${avg.toFixed(2)}`);
        const variance = runs.reduce((prev, current) => prev + (current - avg) * (current - avg));
        const sd = Math.sqrt(variance / (runs.length - 1));
        Logger.verbose('TestRunner.Perf', ` * Runs SD       : ${sd.toFixed(2)}`);
      }
    }
    Logger.verbose('TestRunner.Perf', '***Perf Data End');
  }

  release(): void {
    if (this.profile) {
      this.session.endProfiling();
    }
    this.logPerfData();
  }

  /**
   * create a ModelTestContext object that used in every test cases in the given ModelTest.
   */
  static async create(modelTest: Test.ModelTest, profile: boolean): Promise<ModelTestContext> {
    if (this.initializing) {
      throw new Error('cannot create a ModelTestContext object when the previous creation is not done');
    }

    try {
      this.initializing = true;

      const initStart = now();
      const session = await initializeSession(modelTest.modelUrl, modelTest.backend!, profile, this.cache);
      const initEnd = now();

      for (const testCase of modelTest.cases) {
        await loadTensors(session, testCase, this.cache);
      }

      return new ModelTestContext(
          session,
          modelTest.backend!,
          {init: initEnd - initStart, firstRun: -1, runs: [], count: 0},
          profile,
      );
    } finally {
      this.initializing = false;
    }
  }

  /**
   * set the global file cache for looking up model and tensor protobuf files.
   */
  static setCache(cache: Test.FileCache): void {
    const keys = Object.keys(cache);
    Logger.info('TestRunner', `Setting up file cache... Entry count: ${keys.length}.`);
    for (const key of keys) {
      this.cache[key] = base64toBuffer(cache[key]);
    }
  }

  private static initializing = false;
  private static cache: FileCacheBuffer = {};
}

export declare namespace ModelTestContext {
  export interface ModelTestPerfData {
    init: number;
    firstRun: number;
    runs: number[];
    count: number;
  }
}

export class TensorResultValidator {
  private readonly absoluteThreshold: number;
  private readonly relativeThreshold: number;
  private readonly maxFloatValue: number = 3.4028234663852886e+38;

  private static isHalfFloat: boolean|undefined;

  constructor(backend: string) {
    if (backend === 'cpu') {
      this.absoluteThreshold = CPU_THRESHOLD_ABSOLUTE_ERROR;
      this.relativeThreshold = CPU_THRESHOLD_RELATIVE_ERROR;
    } else if (backend === 'webgl') {
      if (TensorResultValidator.isHalfFloat === undefined) {
        TensorResultValidator.isHalfFloat = !createWebGLContext(ort.env.webgl.contextId).isRenderFloat32Supported;
      }
      if (TensorResultValidator.isHalfFloat) {
        this.maxFloatValue = 65504;
        this.absoluteThreshold = WEBGL_HALF_FLOAT_THRESHOLD_ABSOLUTE_ERROR;
        this.relativeThreshold = WEBGL_HALF_FLOAT_THRESHOLD_RELATIVE_ERROR;
      } else {
        this.absoluteThreshold = WEBGL_THRESHOLD_ABSOLUTE_ERROR;
        this.relativeThreshold = WEBGL_THRESHOLD_RELATIVE_ERROR;
      }
    } else if (backend === 'wasm') {
      this.absoluteThreshold = WASM_THRESHOLD_ABSOLUTE_ERROR;
      this.relativeThreshold = WASM_THRESHOLD_RELATIVE_ERROR;
    } else if (backend === 'onnxruntime') {
      this.absoluteThreshold = ONNXRUNTIME_THRESHOLD_ABSOLUTE_ERROR;
      this.relativeThreshold = ONNXRUNTIME_THRESHOLD_RELATIVE_ERROR;
    } else {
      throw new Error(`backend not supported: ${backend}`);
    }
  }

  checkTensorResult(actual: Tensor[], expected: Tensor[]): void {
    // check output size
    expect(actual.length, 'size of output tensors').to.equal(expected.length);

    // compare output one-by-one
    for (let i = 0; i < actual.length; ++i) {
      const match = this.areEqual(actual[i], expected[i]);
      if (!match) {
        Logger.error(
            'TestRunner',
            `Tensor mismatch: \nACTUAL: type=${actual[i].type}; dims=[${actual[i].dims}]; data=[${
                actual[i].data}]\nEXPECT: type=${expected[i].type}; dims=[${expected[i].dims}]; data=[${
                expected[i].data}]`);
      }
      expect(match, 'tensor data should match').to.be.true;
    }
  }

  checkApiTensorResult(actual: ort.Tensor[], expected: ort.Tensor[]): void {
    this.checkTensorResult(actual.map(toInternalTensor), expected.map(toInternalTensor));
  }

  checkNamedTensorResult(actual: Record<string, ort.Tensor>, expected: Test.NamedTensor[]): void {
    // check output size
    expect(Object.getOwnPropertyNames(actual).length, 'size of output tensors').to.equal(expected.length);

    // check output mapping
    for (const expectedOneOutput of expected) {
      expect(actual, 'keys of output tensors').to.contain.keys(expectedOneOutput.name);
    }

    this.checkApiTensorResult(expected.map(i => actual[i.name]!), expected);
  }

  // This function check whether 2 tensors should be considered as 'match' or not
  areEqual(actual: Tensor, expected: Tensor): boolean {
    if (!actual || !expected) {
      return false;
    }
    if (!actual.dims || !expected.dims) {
      return false;
    }

    const actualDims = actual.dims;
    const actualType = actual.type;
    const expectedDims = expected.dims;
    const expectedType = expected.type;

    if (actualType !== expectedType) {
      return false;
    }
    if (actualDims.length !== expectedDims.length) {
      return false;
    }

    for (let i = 0; i < actualDims.length; i++) {
      if (actualDims[i] !== expectedDims[i]) {
        return false;
      }
    }

    switch (actualType) {
      case 'string':
        return this.strictEqual(actual.stringData, expected.stringData);

      case 'float32':
      case 'float64':
        return this.floatEqual(
            actual.numberData as number[] | Float32Array | Float64Array,
            expected.numberData as number[] | Float32Array | Float64Array);

      case 'uint8':
      case 'int8':
      case 'uint16':
      case 'int16':
      case 'int32':
      case 'uint32':
      case 'bool':
        return this.integerEqual(
            actual.numberData as number[] | Uint8Array | Int8Array | Uint16Array | Int16Array | Uint32Array |
                Int32Array,
            expected.numberData as number[] | Uint8Array | Int8Array | Uint16Array | Int16Array | Uint32Array |
                Int32Array);

      default:
        throw new Error('type not implemented or not supported');
    }
  }
  strictEqual<T>(actual: T, expected: T): boolean {
    try {
      expect(actual).to.deep.equal(expected);
      return true;
    } catch {
      return false;
    }
  }
  floatEqual(actual: number[]|Float32Array|Float64Array, expected: number[]|Float32Array|Float64Array): boolean {
    if (actual.length !== expected.length) {
      return false;
    }

    for (let i = actual.length - 1; i >= 0; i--) {
      const a = actual[i];
      let b = expected[i];

      if (a === b) {
        continue;  // exact the same value, treat as equal
      }

      // check for NaN
      //
      if (Number.isNaN(a) && Number.isNaN(b)) {
        continue;  // 2 numbers are NaN, treat as equal
      }
      if (Number.isNaN(a) || Number.isNaN(b)) {
        Logger.error('Validator', `a or b isNan -- index:${i}: actual=${actual[i]},expected=${expected[i]}`);
        return false;  // one is NaN and the other is not
      }

      // check for Infinity
      //
      if (!Number.isFinite(a) || !Number.isFinite(b)) {
        Logger.error('Validator', `a or b is Infinity -- index:${i}: actual=${actual[i]},expected=${expected[i]}`);
        return false;  // at least one is Infinity and the other is not or their sign is different
      }

      // normalize value of b
      b = Math.max(Math.min(expected[i], this.maxFloatValue), -this.maxFloatValue);

      // Comparing 2 float numbers: (Suppose a >= b)
      //
      // if ( a - b < ABSOLUTE_ERROR || 1.0 < a / b < RELATIVE_ERROR)
      //   test pass
      // else
      //   test fail
      // endif
      //
      if (Math.abs(actual[i] - expected[i]) < this.absoluteThreshold) {
        continue;  // absolute error check pass
      }
      if (a !== 0 && b !== 0 && a / b < this.relativeThreshold && b / a < this.relativeThreshold) {
        continue;  // relative error check pass
      }

      // if code goes here, it means both (abs/rel) check failed.
      Logger.error('Validator', `abs/rel check failed-- index:${i}: actual=${actual[i]},expected=${expected[i]}`);
      return false;
    }

    return true;
  }
  integerEqual(
      actual: number[]|Uint8Array|Int8Array|Uint16Array|Int16Array|Uint32Array|Int32Array,
      expected: number[]|Uint8Array|Int8Array|Uint16Array|Int16Array|Uint32Array|Int32Array): boolean {
    if (actual.length !== expected.length) {
      return false;
    }

    for (let i = actual.length - 1; i >= 0; i--) {
      if (actual[i] !== expected[i]) {
        return false;
      }
    }

    return true;
  }
}

/**
 * run a single model test case. the inputs/outputs tensors should already been prepared.
 */
export async function runModelTestSet(
    context: ModelTestContext, testCase: Test.ModelTestCase, testName: string): Promise<void> {
  Logger.verbose('TestRunner', `Start to run test data from folder: ${testName}/${testCase.name}`);
  Logger.verbose('TestRunner', `Start to run test data from folder: ${testCase.name}`);
  const validator = new TensorResultValidator(context.backend);
  try {
    const feeds: Record<string, ort.Tensor> = {};
    testCase.inputs!.forEach((tensor, i) => feeds[context.session.inputNames[i]] = tensor);
    const start = now();
    const outputs = await context.session.run(feeds);
    const end = now();
    if (context.perfData.count === 0) {
      context.perfData.firstRun = end - start;
    } else {
      context.perfData.runs.push(end - start);
    }
    context.perfData.count++;

    Logger.verbose('TestRunner', `Finished running model from file: ${testCase.name}`);
    Logger.verbose('TestRunner', ' Stats:');
    Logger.verbose('TestRunner', `  Input(s): ${testCase.inputs!.length}`);
    testCase.inputs!.forEach(i => {
      Logger.verbose('TestRunner', `   '${i.name}': ${i.type}[${i.dims.join(',')}]`);
    });
    Logger.verbose('TestRunner', `  Output(s): ${outputs.size}`);
    for (const name in outputs) {
      if (Object.hasOwnProperty.call(outputs, name)) {
        const tensor = outputs[name];
        Logger.verbose('TestRunner', `   '${name}': ${tensor.type}[${tensor.dims.join(',')}]`);
      }
    }

    validator.checkNamedTensorResult(outputs, testCase.outputs!);

    Logger.verbose('TestRunner', '  Result: PASS');
  } catch (e) {
    Logger.error('TestRunner', '  Result: FAILED');
    Logger.error('TestRunner', `Failed to run test data from folder: ${testCase.name}. Error: ${inspect(e)}`);
    throw e;
  }
}

function initializeOperator(
    sessionHandler: SessionHandler, opType: string, attributeValues: readonly Test.AttributeValue[],
    opsetImports: readonly Test.OperatorTestOpsetImport[]): Operator {
  const attributes = new Attribute(undefined);
  attributeValues.forEach(value => attributes.set(value.name, value.type, value.data));
  const graph = createMockGraph(opType, attributes);
  return sessionHandler.resolve(graph.getNodes()[0], opsetImports, graph);
}

/**
 * a OpTestContext object contains all states in a OpTest
 */
export class OpTestContext {
  static profiler = Profiler.create();

  readonly backendHint: string;
  sessionHandler: SessionHandler;
  inferenceHandler: InferenceHandler;

  constructor(protected opTest: Test.OperatorTest) {
    this.backendHint = opTest.backend === 'webgl' ? 'webgl' : 'cpu';
  }
  createOperator(): Operator {
    return initializeOperator(
        this.sessionHandler, this.opTest.operator, this.opTest.attributes,
        this.opTest.opsets ?? [{domain: '', version: 7}]);
  }

  dispose(): void {
    this.inferenceHandler.dispose();
    this.sessionHandler.dispose();
  }

  async init(): Promise<void> {
    const backend = await resolveBackend(this.backendHint);
    this.sessionHandler = backend.createSessionHandler({profiler: OpTestContext.profiler});
    this.inferenceHandler = this.sessionHandler.createInferenceHandler();
  }
}


function createTensor(dims: number[], type: Tensor.DataType, data: number[]): Tensor {
  const tensor = new Tensor(dims, type);
  for (let i = 0; i < data.length; ++i) {
    tensor.data[i] = data[i];
  }
  return tensor;
}

async function runOpTestcase(
    inferenceHandler: InferenceHandler, operator: Operator, testcase: Test.OperatorTestCase,
    validator: TensorResultValidator): Promise<void> {
  testcase.inputs.forEach((input, i) => {
    Logger.verbose('TestOpRunner', `   Input '${i}': ${input.type}[${input.dims.join(',')}]`);
  });
  const inputTensors =
      testcase.inputs.map(input => createTensor(input.dims, input.type as Tensor.DataType, input.data));

  let results = operator.run(inferenceHandler, inputTensors);
  if ('then' in results) {
    results = await results;
  }

  results.forEach((output, i) => {
    Logger.verbose('TestOpRunner', `  Result'${i}': ${output.type}[${output.dims.join(',')}]`);
  });
  const expectedTensors =
      testcase.outputs.map(output => createTensor(output.dims, output.type as Tensor.DataType, output.data));
  validator.checkTensorResult(results, expectedTensors);
}

/**
 * run a single operator test case.
 */
export async function runOpTest(testcase: Test.OperatorTestCase, context: OpTestContext): Promise<void> {
  await runOpTestcase(
      context.inferenceHandler, context.createOperator(), testcase, new TensorResultValidator(context.backendHint));
}
