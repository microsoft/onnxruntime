// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {createRequire} = require('node:module');
const {execFileSync} = require('node:child_process');
const {pathToFileURL} = require('node:url');

async function main() {
  const [archive, versionFile] = process.argv.slice(2).map((value) => path.resolve(value));
  const directory = fs.mkdtempSync(path.join(process.env.TEST_TMPDIR, 'common-npm-consumer-'));
  const node = process.env.JS_BINARY__NODE_BINARY;
  const npm = process.platform === 'win32' ?
    path.join(path.dirname(node), 'node_modules/npm/bin/npm-cli.js') :
    path.resolve(path.dirname(node), '../lib/node_modules/npm/bin/npm-cli.js');
  fs.writeFileSync(path.join(directory, 'package.json'), '{"name":"ort-common-consumer","private":true}');
  fs.writeFileSync(path.join(directory, 'empty.npmrc'), '');
  execFileSync(node, [
    npm, 'install', '--offline', '--ignore-scripts', '--no-audit', '--no-fund', '--no-package-lock', archive,
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
  const root = path.join(directory, 'node_modules/onnxruntime-common');
  const metadata = JSON.parse(fs.readFileSync(path.join(root, 'package.json'), 'utf8'));
  const version = fs.readFileSync(versionFile, 'utf8').trim();
  assert.equal(metadata.version, version);
  assert.equal(metadata.scripts, undefined);
  for (const name of ['dist/cjs/index.js', 'dist/cjs/index.d.ts', 'dist/esm/index.js',
    'LICENSE', 'ThirdPartyNotices.txt', 'README.md']) {
    assert.ok(fs.existsSync(path.join(root, name)), `Installed common package is missing ${name}.`);
  }
  const requireFromConsumer = createRequire(path.join(directory, 'consumer.cjs'));
  assert.equal(requireFromConsumer('onnxruntime-common').env.versions.common, version);
  const esm = await import(pathToFileURL(path.join(root, 'dist/esm/index.js')).href);
  assert.equal(esm.env.versions.common, version);
  const ts = createRequire(path.resolve('bazel/node/package.json'))('typescript');
  const source = path.join(directory, 'consumer.ts');
  fs.writeFileSync(source, [
    'import {InferenceSession, Tensor} from "onnxruntime-common";',
    'const input: Tensor = new Tensor("float32", new Float32Array(2), [2]);',
    'async function run(): Promise<void> {',
    '  const session: InferenceSession = await InferenceSession.create("model.onnx");',
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
  console.log('Offline common npm install, CJS+ESM APIs, and declarations passed.');
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
