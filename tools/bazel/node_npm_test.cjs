// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {createRequire} = require('node:module');
const {execFileSync} = require('node:child_process');
const {pathToFileURL} = require('node:url');

async function main() {
  const [archive, commonArchive, model, versionFile] =
      process.argv.slice(2).map((value) => path.resolve(value));
  const directory = fs.mkdtempSync(path.join(process.env.TEST_TMPDIR, 'npm-consumer-'));
  const node = process.env.JS_BINARY__NODE_BINARY;
  assert.ok(node, 'Run this consumer test through its js_test target to select the pinned Node toolchain.');
  const npm = process.platform === 'win32' ?
    path.join(path.dirname(node), 'node_modules/npm/bin/npm-cli.js') :
    path.resolve(path.dirname(node), '../lib/node_modules/npm/bin/npm-cli.js');
  assert.ok(fs.existsSync(npm), `Pinned Node toolchain is missing npm CLI ${npm}; enable include_npm on js_test.`);
  const config = path.join(directory, 'empty.npmrc');
  fs.writeFileSync(config, '');
  fs.writeFileSync(path.join(directory, 'package.json'), JSON.stringify({name: 'ort-offline-consumer', private: true}));
  execFileSync(node, [
    npm, 'install', '--offline', '--ignore-scripts', '--no-audit', '--no-fund', '--no-package-lock',
    commonArchive, archive,
  ], {
    cwd: directory,
    stdio: 'inherit',
    env: {
      ...process.env,
      HOME: directory,
      USERPROFILE: directory,
      npm_config_cache: path.join(directory, 'cache'),
      npm_config_userconfig: config,
      npm_config_globalconfig: path.join(directory, 'global.npmrc'),
      npm_config_update_notifier: 'false',
    },
  });
  const consumerRequire = createRequire(path.join(directory, 'consumer.cjs'));
  const packageRoot = path.join(directory, 'node_modules/onnxruntime-node');
  const metadata = JSON.parse(fs.readFileSync(path.join(packageRoot, 'package.json'), 'utf8'));
  assert.equal(metadata.version, fs.readFileSync(versionFile, 'utf8').trim());
  assert.equal(metadata.scripts, undefined, 'Production npm package must not run installer or legacy build scripts.');
  assert.deepEqual(metadata.binary.napi_versions, [6]);
  assert.deepEqual(metadata.os, [process.platform]);
  assert.deepEqual(metadata.cpu, [process.arch]);
  assert.ok(['cpu', 'cuda'].includes(metadata.onnxruntimeBazelExecutionProviders.at(-1)));
  assert.deepEqual(metadata.bundledDependencies, ['onnxruntime-common']);
  const files = fs.readdirSync(packageRoot, {recursive: true});
  assert.ok(!files.some((name) => /(^|[/\\])(test|testdata|unittests|script)([/\\]|$)/.test(name)),
      'Production package must not include tests, fixtures, or legacy scripts.');
  assert.ok(files.includes(path.join('dist', 'index.d.ts')), 'Compiler-produced API declarations must be shipped.');
  for (const name of ['LICENSE', 'ThirdPartyNotices.txt', 'README.md']) {
    assert.ok(files.includes(name), `Production Node package must include ${name}.`);
  }
  const nativeRoot = path.join('bin', 'napi-v6', process.platform, process.arch);
  const nativeFiles = files.filter((name) => name.startsWith(nativeRoot + path.sep));
  const expectedNativeFiles = metadata.onnxruntimeBazelExecutionProviders.includes('cuda') ? 4 : 3;
  assert.equal(
      nativeFiles.length, expectedNativeFiles,
      `Expected exactly ${expectedNativeFiles} source-built native payloads under ${nativeRoot}.`);
  if (expectedNativeFiles === 4) {
    assert.ok(nativeFiles.some((name) => name.endsWith('libonnxruntime_providers_cuda.so')));
  }
  const nativePrefix = path.join('bin', 'napi-v6');
  const platformRoot = path.join(nativePrefix, process.platform);
  assert.ok(!files.some((name) =>
    name.startsWith(nativePrefix + path.sep) &&
    name !== platformRoot &&
    name !== nativeRoot &&
    !name.startsWith(nativeRoot + path.sep)),
      'Single-platform package must not contain another platform or architecture.');
  const commonRoot = path.join(directory, 'node_modules/onnxruntime-common');
  const commonMetadata = JSON.parse(fs.readFileSync(path.join(commonRoot, 'package.json'), 'utf8'));
  assert.equal(commonMetadata.version, metadata.version);
  assert.equal(commonMetadata.scripts, undefined);
  const commonRequire = consumerRequire('onnxruntime-common');
  assert.equal(commonRequire.env.versions.common, metadata.version);
  const commonImport = await import(pathToFileURL(path.join(commonRoot, 'dist/esm/index.js')).href);
  assert.equal(commonImport.env.versions.common, metadata.version);
  const ts = createRequire(path.resolve('bazel/node/package.json'))('typescript');
  const source = path.join(directory, 'consumer.ts');
  fs.writeFileSync(source, [
    'import {InferenceSession, Tensor} from "onnxruntime-node";',
    'const input: Tensor = new Tensor("float32", new Float32Array(6), [2, 3]);',
    'async function run(): Promise<void> {',
    '  const session: InferenceSession = await InferenceSession.create("model.onnx");',
    '  await session.run({A: input, B: input});',
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
  const ort = consumerRequire('onnxruntime-node');
  assert.equal(ort.env.versions.node, metadata.version);
  assert.equal(ort.env.versions.common, metadata.version);
  const session = await ort.InferenceSession.create(model, {executionProviders: ['cpu']});
  try {
    const result = await session.run({
      A: new ort.Tensor('float32', Float32Array.from([1, 2, 3, 4, 5, 6]), [2, 3]),
      B: new ort.Tensor('float32', Float32Array.from([10, 20, 30, 40, 50, 60]), [2, 3]),
    });
    assert.deepEqual(result.C.dims, [2, 3]);
    assert.deepEqual(Array.from(result.C.data), [11, 22, 33, 44, 55, 66]);
  } finally {
    await session.release();
  }
  console.log('Offline common/Node npm install, CJS+ESM APIs, declarations, and native CPU Add inference passed.');
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
