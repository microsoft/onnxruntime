const HELP = `
  Call onnx_test_runner to test WebGPU EP.

  Usage: node test_webgpu.js [options]

    Options:
        -h              Print this help message.
        -t=<path>       Path of the test data folder (eg. "../../../js/test/data/node")
        -v              Verbose mode.
        -m=<list>       ';' separated list of test names (eg. test_abs)
`;

const DEFAULT_TESTS = [
    'test_abs',
];

const path = require('path');
const fs = require('fs');
const { spawnSync } = require('child_process');

const ONNX_TEST_RUNNER_FILENAME = path.join(__dirname,
    'onnx_test_runner' + (process.platform === 'win32' ? '.exe' : ''));

if (process.argv.includes('-h')) {
    console.log(HELP);
    process.exit(0);
}

const VERBOSE = process.argv.includes('-v');
let test_data_path = process.argv.find(arg => arg.startsWith('-t='));
if (!test_data_path) {
    test_data_path = path.join(__dirname, (process.platform === 'win32' ? '../' : '') + '../../../js/test/data/node');
} else {
    test_data_path = test_data_path.substring(3);
}

const test_models = [];
const test_model_list = process.argv.find(arg => arg.startsWith('-m='));
if (test_model_list) {
    test_model_list.substring(3).split(';').forEach(test_model => {
        test_models.push(test_model);
    });
}
const tests = new Set(test_model_list ? test_models : DEFAULT_TESTS);
const test_cases = [];
fs.readdirSync(test_data_path, { withFileTypes: true }).forEach(dirent => {
    if (dirent.isDirectory()) {
        const opset = dirent.name;
        fs.readdirSync(path.join(test_data_path, opset), { withFileTypes: true }).forEach(dirent => {
            if (dirent.isDirectory()) {
                const name = dirent.name;
                if (tests.has(name)) {
                    test_cases.push(path.join(test_data_path, opset, name));
                }
            }
        });
    }
});

let passed = [];
let not_implemented = [];
let failed = [];
test_cases.forEach(test_case => {
    process.stdout.write(`Running test case: "${test_case}"...`);
    const args = [
        '-e', 'webgpu', '-C', '"session.disable_cpu_ep_fallback|1"', test_case,
    ];
    if (VERBOSE) {
        args.unshift('-v');
    }
    const p = spawnSync(ONNX_TEST_RUNNER_FILENAME, args, { shell: true, stdio: ['ignore', 'pipe', 'pipe'] });
    if (p.status !== 0) {
        process.stdout.write('Failed\n');
        failed.push(test_case);
    } else if (!p.stdout.toString().includes('Not implemented: 0')) {
        process.stdout.write('Not Implemented\n');
        not_implemented.push(test_case);
    } else {
        process.stdout.write('OK\n');
        passed.push(test_case);
    }
});

console.log(`\n${passed.length} tests passed.`);
console.log(`\n${not_implemented.length} tests not implemented:`);
not_implemented.slice(0, 3).forEach(test_case => {
    console.log(`  ${test_case}`);
});
if (not_implemented.length > 3) {
    console.log(`  ...`);
}
console.log(`\n${failed.length} tests failed:`);
failed.slice(0, 3).forEach(test_case => {
    console.log(`  ${test_case}`);
});
if (failed.length > 3) {
    console.log(`  ...`);
}
