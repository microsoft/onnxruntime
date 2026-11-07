// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {createRequire} = require('node:module');
const {execFileSync} = require('node:child_process');

async function main() {
  const [archive, model, versionFile, variant] = process.argv.slice(2);
  const directory = fs.mkdtempSync(path.join(process.env.TEST_TMPDIR, `web-${variant}-consumer-`));
  const node = process.env.JS_BINARY__NODE_BINARY;
  const npm = process.platform === 'win32' ?
    path.join(path.dirname(node), 'node_modules/npm/bin/npm-cli.js') :
    path.resolve(path.dirname(node), '../lib/node_modules/npm/bin/npm-cli.js');
  fs.writeFileSync(path.join(directory, 'package.json'), '{"name":"ort-web-offline-consumer","private":true}');
  fs.writeFileSync(path.join(directory, 'empty.npmrc'), '');
  execFileSync(node, [
    npm, 'install', '--offline', '--ignore-scripts', '--no-audit', '--no-fund', '--no-package-lock',
    path.resolve(archive),
  ], {
    cwd: directory,
    stdio: 'inherit',
    env: {
      ...process.env,
      HOME: directory,
      USERPROFILE: directory,
      npm_config_cache: path.join(directory, 'cache'),
      npm_config_userconfig: path.join(directory, 'empty.npmrc'),
      npm_config_globalconfig: path.join(directory, 'global.npmrc'),
      npm_config_update_notifier: 'false',
    },
  });
  const root = path.join(directory, 'node_modules/onnxruntime-web');
  const metadata = JSON.parse(fs.readFileSync(path.join(root, 'package.json'), 'utf8'));
  assert.equal(metadata.version, fs.readFileSync(versionFile, 'utf8').trim());
  assert.equal(metadata.onnxruntimeBazelVariant, variant);
  assert.deepEqual(metadata.onnxruntimeBazelFeatures, {
    threads: false, proxy: false, webgpu: false, webnn: false, jsep: false,
  });
  assert.equal(metadata.scripts, undefined);
  for (const name of ['dist/ort.js', 'dist/ort-wasm.mjs', 'dist/ort-wasm.wasm', 'types/index.d.ts',
    'LICENSE', 'ThirdPartyNotices.txt', 'README.md']) {
    assert.ok(fs.existsSync(path.join(root, name)), `Installed Web package is missing ${name}.`);
  }
  const ts = createRequire(path.resolve('bazel/node/package.json'))('typescript');
  const source = path.join(directory, 'consumer.ts');
  fs.writeFileSync(source, [
    'import {InferenceSession, Tensor} from "onnxruntime-web";',
    'const input: Tensor = new Tensor("float32", new Float32Array(6), [3, 2]);',
    'async function run(): Promise<void> {',
    '  const session: InferenceSession = await InferenceSession.create(new Uint8Array());',
    '  await session.run({X: input});',
    '}',
    'void run;',
  ].join('\n'));
  const program = ts.createProgram([source], {
    target: ts.ScriptTarget.ES2020,
    module: ts.ModuleKind.CommonJS,
    moduleResolution: ts.ModuleResolutionKind.Node10,
    strict: true,
    noEmit: true,
    lib: ['lib.es2020.d.ts', 'lib.esnext.float16.d.ts', 'lib.dom.d.ts'],
    types: ['node'],
    typeRoots: [path.resolve('bazel/node/node_modules/@types')],
  });
  const errors = ts.getPreEmitDiagnostics(program).filter((item) => item.category === ts.DiagnosticCategory.Error);
  assert.equal(errors.length, 0, ts.formatDiagnosticsWithColorAndContext(errors, {
    getCurrentDirectory: () => process.cwd(),
    getCanonicalFileName: (name) => name,
    getNewLine: () => '\n',
  }));
  const ort = createRequire(path.join(directory, 'consumer.cjs'))('onnxruntime-web');
  ort.env.wasm.wasmBinary = fs.readFileSync(path.join(root, 'dist/ort-wasm.wasm'));
  const session = await ort.InferenceSession.create(fs.readFileSync(model), {executionProviders: ['wasm']});
  try {
    const {Y} = await session.run({
      X: new ort.Tensor('float32', Float32Array.from([1, 2, 3, 4, 5, 6]), [3, 2]),
    });
    assert.deepEqual(Y.dims, [3, 2]);
    assert.deepEqual(Array.from(Y.data), [1, 4, 9, 16, 25, 36]);
  } finally {
    await session.release();
  }
  console.log(`Offline ${variant} Web npm install, declarations, metadata, and WASM inference passed.`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
