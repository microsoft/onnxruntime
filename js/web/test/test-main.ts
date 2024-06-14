// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Load onnxruntime-common and testdata-config.
// NOTE: this need to be called before import any other library.
import * as ort from 'onnxruntime-common';

const ORT_WEB_TEST_CONFIG = require('./testdata-config.json') as Test.Config;

import * as platform from 'platform';

import {Logger} from '../lib/onnxjs/instrument';

import {Test} from './test-types';

if (ORT_WEB_TEST_CONFIG.model.some(testGroup => testGroup.tests.some(test => test.backend === 'cpu'))) {
  // require onnxruntime-node
  require('../../node');
}

// set flags
Object.assign(ort.env, ORT_WEB_TEST_CONFIG.options.globalEnvFlags);

// Set logging configuration
for (const logConfig of ORT_WEB_TEST_CONFIG.log) {
  Logger.set(logConfig.category, logConfig.config);
}

import {ModelTestContext, OpTestContext, ProtoOpTestContext, runModelTestSet, runOpTest} from './test-runner';
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
  if (!test.platformCondition) {
    return false;
  }

  if (!platform.description) {
    throw new Error('failed to check current platform');
  }
  const regex = new RegExp(test.platformCondition);
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
          context = await ModelTestContext.create(test, ORT_WEB_TEST_CONFIG.profile, ORT_WEB_TEST_CONFIG.options);
        });

        after('release session', async () => {
          if (context) {
            await context.release();
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
      const backend = test.backend!;
      const useProtoOpTest = backend !== 'webgl';
      describeTest(`[${backend}]${test.operator} - ${test.name}`, () => {
        let context: ProtoOpTestContext|OpTestContext;

        before('Initialize Context', async () => {
          context = useProtoOpTest ? new ProtoOpTestContext(test, ORT_WEB_TEST_CONFIG.options.sessionOptions) :
                                     new OpTestContext(test);
          await context.init();
          if (ORT_WEB_TEST_CONFIG.profile) {
            if (context instanceof ProtoOpTestContext) {
              context.session.startProfiling();
            } else {
              OpTestContext.profiler.start();
            }
          }
        });

        after('Dispose Context', async () => {
          if (context) {
            if (ORT_WEB_TEST_CONFIG.profile) {
              if (context instanceof ProtoOpTestContext) {
                context.session.endProfiling();
              } else {
                OpTestContext.profiler.stop();
              }
            }
            await context.dispose();
          }
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
