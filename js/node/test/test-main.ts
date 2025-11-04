// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import { NODE_TESTS_ROOT, warmup } from './test-utils';

// require onnxruntime-node.
require('..');

// warmup
//
// for unknown reason, the first call to native InferenceSession::Run() is very slow.
// we need this warmup call so that coming test cases will not fail because of timeout.
warmup();

// unittests
require('./unittests/lib/index');
require('./unittests/lib/inference-session');
require('./unittests/lib/model-metadata');
require('./unittests/lib/tensor');

// API tests
require('./api/simple-api-tests');
require('./api/inference-session-run');
require('./api/worker-test');

// standalone tests
require('./standalone/index');

// Test ONNX spec tests
import { run as runTestRunner } from './test-runner';
describe('ONNX spec tests', () => {
  runTestRunner(NODE_TESTS_ROOT);
});
