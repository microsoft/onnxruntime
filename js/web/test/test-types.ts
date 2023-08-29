// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {Env, InferenceSession, Tensor} from 'onnxruntime-common';

import {Attribute} from '../lib/onnxjs/attribute';
import {Logger} from '../lib/onnxjs/instrument';

export declare namespace Test {
  export interface NamedTensor extends Tensor {
    name: string;
  }

  /**
   * This interface represent a value of Attribute. Should only be used in testing.
   */
  export interface AttributeValue {
    name: string;
    data: Attribute.DataTypeMap[Attribute.DataType];
    type: Attribute.DataType;
  }

  /**
   * This interface represent a value of Tensor. Should only be used in testing.
   */
  export interface TensorValue {
    data: number[];
    dims: number[];
    type: Tensor.Type;
  }

  /**
   * This interface represent a placeholder for an empty tensor. Should only be used in testing.
   */
  interface EmptyTensorValue {
    data: null;
    type: Tensor.Type;
  }

  /**
   * Represent a string to describe the current environment.
   * Used in ModelTest and OperatorTest to determine whether to run the test or not.
   */
  export type PlatformCondition = string;

  export interface ModelTestCase {
    name: string;
    dataFiles: readonly string[];
    inputs?: NamedTensor[];   // value should be populated at runtime
    outputs?: NamedTensor[];  // value should be populated at runtime
  }

  export interface ModelTest {
    name: string;
    modelUrl: string;
    backend?: string;  // value should be populated at build time
    platformCondition?: PlatformCondition;
    cases: readonly ModelTestCase[];
  }

  export interface ModelTestGroup {
    name: string;
    tests: readonly ModelTest[];
  }

  export interface OperatorTestCase {
    name: string;
    inputs: ReadonlyArray<TensorValue|EmptyTensorValue>;
    outputs: ReadonlyArray<TensorValue|EmptyTensorValue>;
  }

  export interface OperatorTestOpsetImport {
    domain: string;
    version: number;
  }

  export type InputShapeDefinition = ReadonlyArray<number|string>;

  export interface OperatorTest {
    name: string;
    operator: string;
    inputShapeDefinitions?: 'none'|'rankOnly'|'static'|ReadonlyArray<InputShapeDefinition|undefined>;
    opset?: OperatorTestOpsetImport;
    backend?: string;  // value should be populated at build time
    platformCondition?: PlatformCondition;
    attributes?: readonly AttributeValue[];
    cases: readonly OperatorTestCase[];
  }

  export interface OperatorTestGroup {
    name: string;
    tests: readonly OperatorTest[];
  }

  // eslint-disable-next-line @typescript-eslint/no-namespace
  export namespace TestList {
    export type TestName = string;
    export interface TestDescription {
      name: string;
      platformCondition: PlatformCondition;
    }
    export type Test = TestName|TestDescription;
  }

  /**
   * The data schema of a testlist file.
   * A testlist should only be applied when running suite test cases (suite0)
   */
  export interface TestList {
    [backend: string]: {[group: string]: readonly TestList.Test[]};
  }

  interface EnvOptions extends Partial<Omit<Env, 'wasm'|'webgl'|'webgpu'>> {
    wasm: Partial<Env.WebAssemblyFlags>;
    webgl: Partial<Env.WebGLFlags>;
    webgpu: Partial<Env.WebGpuFlags>;
  }

  /**
   * Represent ONNX Runtime Web global options
   */
  export interface Options {
    debug?: boolean;
    sessionOptions?: InferenceSession.SessionOptions;
    cpuOptions?: InferenceSession.CpuExecutionProviderOption;
    cpuFlags?: Record<string, unknown>;
    cudaOptions?: InferenceSession.CudaExecutionProviderOption;
    cudaFlags?: Record<string, unknown>;
    wasmOptions?: InferenceSession.WebAssemblyExecutionProviderOption;
    webglOptions?: InferenceSession.WebGLExecutionProviderOption;
    globalEnvFlags?: EnvOptions;
  }

  /**
   * Represent a file cache map that preload the files in prepare stage.
   * The key is the file path and the value is the file content in BASE64.
   */
  export interface FileCache {
    [filePath: string]: string;
  }

  /**
   * The data schema of a test config.
   */
  export interface Config {
    unittest: boolean;
    op: readonly OperatorTestGroup[];
    model: readonly ModelTestGroup[];

    fileCacheUrls?: readonly string[];

    log: ReadonlyArray<{category: string; config: Logger.Config}>;
    profile: boolean;
    options: Options;
  }
}
