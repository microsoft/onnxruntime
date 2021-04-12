// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import * as api from '../lib/api';
import {Backend} from '../lib/api';
import {Attribute} from '../lib/attribute';
import {Logger} from '../lib/instrument';
import {Tensor} from '../lib/tensor';

export declare namespace Test {
  export interface NamedTensor extends api.Tensor {
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
    type: Tensor.DataType;
  }

  /**
   * Represent a string to describe the current environment.
   * Used in ModelTest and OperatorTest to determine whether to run the test or not.
   */
  export type Condition = string;

  export interface ModelTestCase {
    name: string;
    dataFiles: ReadonlyArray<string>;
    inputs?: NamedTensor[];   // value should be populated at runtime
    outputs?: NamedTensor[];  // value should be populated at runtime
  }

  export interface ModelTest {
    name: string;
    modelUrl: string;
    backend?: string;  // value should be populated at build time
    condition?: Condition;
    cases: ReadonlyArray<ModelTestCase>;
  }

  export interface ModelTestGroup {
    name: string;
    tests: ReadonlyArray<ModelTest>;
  }

  export interface OperatorTestCase {
    name: string;
    inputs: ReadonlyArray<TensorValue>;
    outputs: ReadonlyArray<TensorValue>;
  }

  export interface OperatorTestOpsetImport {
    domain: string;
    version: number;
  }

  export interface OperatorTest {
    name: string;
    operator: string;
    opsets?: ReadonlyArray<OperatorTestOpsetImport>;
    backend?: string;  // value should be populated at build time
    condition?: Condition;
    attributes: ReadonlyArray<AttributeValue>;
    cases: ReadonlyArray<OperatorTestCase>;
  }

  export interface OperatorTestGroup {
    name: string;
    tests: ReadonlyArray<OperatorTest>;
  }

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
    [backend: string]: {[group: string]: ReadonlyArray<WhiteList.Test>;};
  }

  /**
   * Represent ONNX.js global options
   */
  export interface Options {
    debug?: boolean;
    cpu?: Backend.CpuOptions;
    webgl?: Backend.WebGLOptions;
    wasm?: Backend.WasmOptions;
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
    op: ReadonlyArray<OperatorTestGroup>;
    model: ReadonlyArray<ModelTestGroup>;

    fileCacheUrls?: ReadonlyArray<string>;

    log: ReadonlyArray<{category: string, config: Logger.Config}>;
    profile: boolean;
    options: Options;
  }
}
