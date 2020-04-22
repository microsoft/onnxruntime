import * as path from 'path';

// unittests
require('./unittests/lib/index');
require('./unittests/lib/inference-session');
require('./unittests/lib/tensor');

// E2E tests
require('./e2e/simple-e2e-tests');
require('./e2e/inference-session-run');

// Test ONNX spec tests
import {run as runTestRunner} from './test-runner';
const NODE_TESTS_ROOT = path.join(__dirname, '../../../cmake/external/onnx/onnx/backend/test/data/node');
describe('ONNX spec tests', () => {
  runTestRunner(NODE_TESTS_ROOT);
});
