// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {createRequire} = require('node:module');
const {execFileSync} = require('node:child_process');

if (process.platform === 'win32') {
  assert.equal(process.env.MSYS2_ARG_CONV_EXCL, '',
      'Windows JS launcher must restore path conversion without changing the top-level Bazel environment.');
}

const [archive, aar] = process.argv.slice(2).map((value) => path.resolve(value));
const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'ort-rn-consumer-'));
try {
  const node = process.env.JS_BINARY__NODE_BINARY;
  assert.ok(node, 'Run through //:react_native_package_consumer to select pinned Node and npm.');
  const npm = process.platform === 'win32' ?
    path.join(path.dirname(node), 'node_modules/npm/bin/npm-cli.js') :
    path.resolve(path.dirname(node), '../lib/node_modules/npm/bin/npm-cli.js');
  fs.writeFileSync(path.join(directory, 'package.json'), '{"name":"ort-rn-consumer","private":true}');
  fs.writeFileSync(path.join(directory, 'empty.npmrc'), '');
  execFileSync(node, [
    npm, 'install', '--offline', '--ignore-scripts', '--legacy-peer-deps',
    '--no-audit', '--no-fund', '--no-package-lock', archive,
  ], {
    cwd: directory,
    stdio: 'inherit',
    env: {
      ...process.env,
      npm_config_cache: path.join(directory, 'cache'),
      npm_config_userconfig: path.join(directory, 'empty.npmrc'),
      npm_config_globalconfig: path.join(directory, 'global.npmrc'),
      npm_config_update_notifier: 'false',
    },
  });
  const root = path.join(directory, 'node_modules/onnxruntime-react-native');
  const metadata = JSON.parse(fs.readFileSync(path.join(root, 'package.json'), 'utf8'));
  const version = fs.readFileSync('VERSION_NUMBER', 'utf8').trim();
  assert.equal(metadata.version, version);
  assert.equal(metadata.scripts, undefined);
  assert.equal(metadata.peerDependencies['react-native'], '0.73.11');
  assert.deepEqual(metadata.onnxruntimeBazelPlatforms, ['android-arm64']);
  assert.deepEqual(fs.readFileSync(path.join(root, metadata.onnxruntimeBazelAar)), fs.readFileSync(aar));
  assert.ok(fs.existsSync(path.join(root, metadata['react-native'])));
  for (const name of ['api.ts', 'backend.ts', 'binding.ts', 'index.ts', 'version.ts']) {
    assert.ok(fs.existsSync(path.join(root, 'lib', name)), `Installed React Native package is missing lib/${name}.`);
  }
  const installed = createRequire(path.join(root, 'consumer.cjs'));
  const config = installed('./react-native.config.js');
  assert.equal(config.dependency.platforms.ios, null);
  assert.equal(config.dependency.platforms.android.packageInstance, 'new OnnxruntimePackage()');
  const gradle = fs.readFileSync(path.join(root, 'android/build.gradle'), 'utf8');
  for (const [dependency, sdkVersion] of [
    ['com.facebook.react:react-android', '0.73.11'],
    ['com.facebook.fbjni:fbjni', '0.5.1'],
  ]) {
    assert.ok(gradle.includes(`add("default", "${dependency}") {\n        version { strictly("${sdkVersion}") }\n    }`));
  }
  const common = installed('onnxruntime-common');
  const tensor = new common.Tensor('float32', Float32Array.from([1, 2]), [2]);
  assert.deepEqual(Array.from(tensor.data), [1, 2]);
  assert.equal(common.env.versions.common, version);
  const ts = createRequire(path.resolve('bazel/node/package.json'))('typescript');
  const source = path.join(directory, 'consumer.ts');
  const reactNativeStub = path.join(directory, 'react-native.d.ts');
  fs.writeFileSync(source, [
    'import {InferenceSession, Tensor} from "onnxruntime-react-native";',
    'const x: Tensor = new Tensor("float32", new Float32Array(2), [2]);',
    'async function run(): Promise<void> {',
    '  const session: InferenceSession = await InferenceSession.create("model.onnx");',
    '  await session.run({X: x});',
    '}',
    'void run;',
  ].join('\n'));
  fs.writeFileSync(reactNativeStub, 'declare module "react-native" { export const NativeModules: any; }\n');
  const program = ts.createProgram([source, reactNativeStub], {
    target: ts.ScriptTarget.ES2020,
    module: ts.ModuleKind.CommonJS,
    moduleResolution: ts.ModuleResolutionKind.Node10,
    strict: true,
    noEmit: true,
    baseUrl: directory,
    paths: {
      'onnxruntime-common': [path.join(root, 'node_modules/onnxruntime-common/lib/index.d.ts')],
      'onnxruntime-react-native': [path.join(root, 'lib/index.ts')],
    },
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
  console.log('Offline npm install, exact AAR, Android autolink metadata, and installed common API passed; no React/device inference.');
} finally {
  fs.rmSync(directory, {recursive: true, force: true});
}
