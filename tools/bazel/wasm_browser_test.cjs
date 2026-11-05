// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

const assert = require('node:assert/strict');
const fs = require('node:fs');
const http = require('node:http');
const path = require('node:path');
const {spawn} = require('node:child_process');

async function main() {
  const browser = process.env.ORT_BAZEL_BROWSER ||
      (process.platform === 'win32' ?
        path.join(process.env['ProgramFiles(x86)'] || 'C:\\Program Files (x86)',
            'Microsoft/Edge/Application/msedge.exe') : undefined);
  assert(browser && fs.existsSync(browser),
      'Chromium/Edge is required; set --test_env=ORT_BAZEL_BROWSER=/absolute/path/to/chromium.');
  const directory = path.resolve(process.argv[2], 'package', 'dist');
  let reportResult;
  const result = new Promise((resolve) => { reportResult = resolve; });
  const files = {
    '/': ['tools/bazel/wasm_browser_test.html', 'text/html'],
    '/mul_1.onnx': ['onnxruntime/test/testdata/mul_1.onnx', 'application/octet-stream'],
    '/matmul_1.onnx': ['onnxruntime/test/testdata/matmul_1.onnx', 'application/octet-stream'],
    '/ort-web.js': [path.join(directory, 'ort.js'), 'text/javascript'],
    '/ort_wasm.mjs': [path.join(directory, 'ort-wasm.mjs'), 'text/javascript'],
    '/ort_wasm.wasm': [path.join(directory, 'ort-wasm.wasm'), 'application/wasm'],
  };
  const server = http.createServer((request, response) => {
    if (request.url === '/result' && request.method === 'POST') {
      let body = '';
      request.setEncoding('utf8');
      request.on('data', (data) => { body += data; });
      request.on('end', () => {
        response.end('Result received.');
        reportResult(body);
      });
      return;
    }
    const file = files[request.url];
    if (!file) {
      response.writeHead(404);
      response.end('No such test asset.');
      return;
    }
    response.writeHead(200, {'Content-Type': file[1]});
    fs.createReadStream(file[0]).on('error', (error) => response.destroy(error)).pipe(response);
  });
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  const child = spawn(browser, [
      '--headless=new', '--disable-gpu', '--no-first-run', '--no-default-browser-check',
      '--remote-debugging-port=0',
      `--user-data-dir=${path.resolve(process.env.TEST_TMPDIR, 'wasm-browser-profile')}`,
      `http://127.0.0.1:${server.address().port}/`,
    ], {stdio: ['ignore', 'pipe', 'pipe']});
  let timeout;
  try {
    let stderr = '';
    child.stdout.resume();
    child.stderr.on('data', (data) => { stderr += data; });
    const outcome = await Promise.race([
      result,
      new Promise((resolve, reject) => {
        timeout = setTimeout(() => reject(new Error(
            `Browser did not report inference within 180 seconds; inspect browser policy/network errors: ${stderr}`)),
        180000);
        child.once('error', reject);
        child.once('exit', (code) => reject(new Error(`Browser exited before reporting inference (${code}): ${stderr}`)));
      }),
    ]);
    assert.equal(outcome, 'passed', `Browser inference failed: ${outcome}\n${stderr}`);
    console.log('Browser executed source-built WASM/frontend inference with verified shape and all output values.');
  } finally {
    clearTimeout(timeout);
    child.kill();
    server.closeAllConnections();
    await new Promise((resolve) => server.close(resolve));
  }
}
main().catch((error) => { console.error(error); process.exitCode = 1; });
