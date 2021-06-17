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
if (options.globalEnvFlags) {
  const flags = options.globalEnvFlags;
  if (flags.logLevel !== undefined) {
    ort.env.logLevel = flags.logLevel;
  }
  if (flags.webgl?.contextId !== undefined) {
    ort.env.webgl.contextId = flags.webgl.contextId;
  }
  if (flags.webgl?.matmulMaxBatchSize !== undefined) {
    ort.env.webgl.matmulMaxBatchSize = flags.webgl.matmulMaxBatchSize;
  }
  if (flags.webgl?.textureCacheMode !== undefined) {
    ort.env.webgl.textureCacheMode = flags.webgl.textureCacheMode;
  }
  if (flags.webgl?.pack !== undefined) {
    ort.env.webgl.pack = flags.webgl.pack;
  }
  if (flags.wasm?.numThreads !== undefined) {
    ort.env.wasm.numThreads = flags.wasm.numThreads;
  }
  if (flags.wasm?.simd !== undefined) {
    ort.env.wasm.simd = flags.wasm.simd;
  }
  if (flags.wasm?.initTimeout !== undefined) {
    ort.env.wasm.initTimeout = flags.wasm.initTimeout;
  }
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
