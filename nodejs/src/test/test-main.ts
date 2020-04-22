import {NODE_TESTS_ROOT} from './test-utils';

// unittests
require('./unittests/lib/index');
require('./unittests/lib/inference-session');
require('./unittests/lib/tensor');

// E2E tests
require('./e2e/simple-e2e-tests');
require('./e2e/inference-session-run');

// Test ONNX spec tests
import {run as runTestRunner} from './test-runner';
describe('ONNX spec tests', () => {
  runTestRunner(NODE_TESTS_ROOT);
});
