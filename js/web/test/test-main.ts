// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Load onnxruntime-web and testdata-config.
// NOTE: this need to be called before import any other library.
const ort = require('..');
const ORT_WEB_TEST_CONFIG = require('./testdata-config.json') as Test.Config;

import * as platform from 'platform';

import {Logger} from '../lib/onnxjs/instrument';

import {Test} from './test-types';

if (ORT_WEB_TEST_CONFIG.model.some(testGroup => testGroup.tests.some(test => test.backend === 'cpu'))) {
  // require onnxruntime-node
  require('../../node');
}

// set flags
const options = ORT_WEB_TEST_CONFIG.options;
if (options.debug !== undefined) {
  ort.env.debug = options.debug;
}
if (options.globalEnvFlags && options.globalEnvFlags.logLevel !== undefined) {
  ort.env.logLevel = options.globalEnvFlags.logLevel;
}
if (ort.env.webgl && options.globalEnvFlags && options.globalEnvFlags.webgl &&
    options.globalEnvFlags.webgl.contextId !== undefined) {
  ort.env.webgl.contextId = options.globalEnvFlags.webgl.contextId;
}
if (ort.env.webgl && options.globalEnvFlags && options.globalEnvFlags.webgl &&
    options.globalEnvFlags.webgl.matmulMaxBatchSize !== undefined) {
  ort.env.webgl.matmulMaxBatchSize = options.globalEnvFlags.webgl.matmulMaxBatchSize;
}
if (ort.env.webgl && options.globalEnvFlags && options.globalEnvFlags.webgl &&
    options.globalEnvFlags.webgl.textureCacheMode !== undefined) {
  ort.env.webgl.textureCacheMode = options.globalEnvFlags.webgl.textureCacheMode;
}
if (ort.env.webgl && options.globalEnvFlags && options.globalEnvFlags.webgl &&
    options.globalEnvFlags.webgl.pack !== undefined) {
  ort.env.webgl.pack = options.globalEnvFlags.webgl.pack;
}
if (ort.env.wasm && options.globalEnvFlags && options.globalEnvFlags.wasm &&
    options.globalEnvFlags.wasm.numThreads !== undefined) {
  ort.env.wasm.numThreads = options.globalEnvFlags.wasm.numThreads;
}
if (ort.env.wasm && options.globalEnvFlags && options.globalEnvFlags.wasm &&
    options.globalEnvFlags.wasm.initTimeout !== undefined) {
  ort.env.wasm.initTimeout = options.globalEnvFlags.wasm.initTimeout;
}

// Set logging configuration
for (const logConfig of ORT_WEB_TEST_CONFIG.log) {
  Logger.set(logConfig.category, logConfig.config);
}

import {ModelTestContext, OpTestContext, runModelTestSet, runOpTest} from './test-runner';
import {readJsonFile} from './test-shared';

// Unit test
if (ORT_WEB_TEST_CONFIG.unittest) {
  require('./unittests');
}

// Set file cache
if (ORT_WEB_TEST_CONFIG.fileCacheUrls) {
  before('prepare file cache', async () => {
    const allJsonCache = await Promise.all(ORT_WEB_TEST_CONFIG.fileCacheUrls!.map(readJsonFile)) as Test.FileCache[];
    for (const cache of allJsonCache) {
      ModelTestContext.setCache(cache);
    }
  });
}

function shouldSkipTest(test: Test.ModelTest|Test.OperatorTest) {
  if (!test.cases || test.cases.length === 0) {
    return true;
  }
  if (!test.condition) {
    return false;
  }

  if (!platform.description) {
    throw new Error('failed to check current platform');
  }
  const regex = new RegExp(test.condition);
  return !regex.test(platform.description);
}

// ModelTests
for (const group of ORT_WEB_TEST_CONFIG.model) {
  describe(`#ModelTest# - ${group.name}`, () => {
    for (const test of group.tests) {
      const describeTest = shouldSkipTest(test) ? describe.skip : describe;
      describeTest(`[${test.backend}] ${test.name}`, () => {
        let context: ModelTestContext;

        before('prepare session', async () => {
          context = await ModelTestContext.create(test, ORT_WEB_TEST_CONFIG.profile);
        });

        after('release session', () => {
          if (context) {
            context.release();
          }
        });

        for (const testCase of test.cases) {
          it(testCase.name, async () => {
            await runModelTestSet(context, testCase, test.name);
          });
        }
      });
    }
  });
}

// OpTests
for (const group of ORT_WEB_TEST_CONFIG.op) {
  describe(`#OpTest# - ${group.name}`, () => {
    for (const test of group.tests) {
      const describeTest = shouldSkipTest(test) ? describe.skip : describe;
      describeTest(`[${test.backend!}]${test.operator} - ${test.name}`, () => {
        let context: OpTestContext;

        before('Initialize Context', async () => {
          context = new OpTestContext(test);
          await context.init();
          if (ORT_WEB_TEST_CONFIG.profile) {
            OpTestContext.profiler.start();
          }
        });

        after('Dispose Context', () => {
          if (ORT_WEB_TEST_CONFIG.profile) {
            OpTestContext.profiler.stop();
          }
          context.dispose();
        });

        for (const testCase of test.cases) {
          it(testCase.name, async () => {
            await runOpTest(testCase, context);
          });
        }
      });
    }
  });
}
