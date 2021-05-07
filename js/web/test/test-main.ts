// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import '../lib/index'; // this need to be the first line

import * as ort from 'onnxruntime-common';
import * as platform from 'platform';

import {Logger} from '../lib/onnxjs/instrument';

import {Test} from './test-types';

const ORT_WEB_TEST_CONFIG = (ort.env as any).ORT_WEB_TEST_DATA as Test.Config;

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
