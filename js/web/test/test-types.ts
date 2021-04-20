// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {InferenceSession, Tensor} from 'onnxruntime-common';

import {WebGLFlags} from '../lib/backend-onnxjs';
import {WebAssemblyFlags} from '../lib/backend-wasm';
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
   * Represent a string to describe the current environment.
   * Used in ModelTest and OperatorTest to determine whether to run the test or not.
   */
  export type Condition = string;

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
    condition?: Condition;
    cases: readonly ModelTestCase[];
  }

  export interface ModelTestGroup {
    name: string;
    tests: readonly ModelTest[];
  }

  export interface OperatorTestCase {
    name: string;
    inputs: readonly TensorValue[];
    outputs: readonly TensorValue[];
  }

  export interface OperatorTestOpsetImport {
    domain: string;
    version: number;
  }

  export interface OperatorTest {
    name: string;
    operator: string;
    opsets?: readonly OperatorTestOpsetImport[];
    backend?: string;  // value should be populated at build time
    condition?: Condition;
    attributes: readonly AttributeValue[];
    cases: readonly OperatorTestCase[];
  }

  export interface OperatorTestGroup {
    name: string;
    tests: readonly OperatorTest[];
  }

  // eslint-disable-next-line @typescript-eslint/no-namespace
  export namespace WhiteList {
    export type TestName = string;
    export interface TestDescription {
      name: string;
      condition: Condition;
    }
    export type Test = TestName|TestDescription;
  }

  /**
   * The data schema of a whitelist file.
   * A whitelist should only be applied when running suite test cases (suite0, suite1)
   */
  export interface WhiteList {
    [backend: string]: {[group: string]: readonly WhiteList.Test[]};
  }

  /**
   * Represent ONNX.js global options
   */
  export interface Options {
    debug?: boolean;
    cpuOptions?: InferenceSession.CpuExecutionProviderOption;
    cpuFlags?: Record<string, unknown>;
    cudaOptions?: InferenceSession.CudaExecutionProviderOption;
    cudaFlags?: Record<string, unknown>;
    wasmOptions?: InferenceSession.WebAssemblyExecutionProviderOption;
    wasmFlags?: WebAssemblyFlags;
    webglOptions?: InferenceSession.WebGLExecutionProviderOption;
    webglFlags?: WebGLFlags;
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
