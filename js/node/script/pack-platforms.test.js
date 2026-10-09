// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { execFileSync } = require('node:child_process');
const { test, before, after } = require('node:test');
const { packPlatforms, findNpmCli } = require('./pack-platforms.js');
const { getNativeBindingPath } = require('./native-directory.js');

const version = '1.32.0';
const libraries = { linux: 'libonnxruntime.so.1', darwin: 'libonnxruntime.1.dylib', win32: 'onnxruntime.dll' };
const writeJson = (filename, value) => fs.writeFileSync(filename, JSON.stringify(value));
let root;
let source;
let archives;
let unpacked;
let originalManifest;

before(() => {
  root = fs.mkdtempSync(path.join(os.tmpdir(), 'ort-packaging-test-'));
  source = path.join(root, 'source');
  fs.mkdirSync(source);
  originalManifest = JSON.stringify({
    name: 'onnxruntime-node',
    version,
    license: 'MIT',
    main: 'dist/index.js',
    scripts: { prepare: 'exit 1', prepack: 'exit 1', postinstall: 'node ./script/install.js' },
  });
  fs.writeFileSync(path.join(source, 'package.json'), originalManifest);
  fs.copyFileSync(path.join(__dirname, '..', '.npmignore'), path.join(source, '.npmignore'));
  fs.mkdirSync(path.join(source, 'dist'));
  fs.writeFileSync(path.join(source, 'dist/index.js'), 'module.exports = {};');
  fs.mkdirSync(path.join(source, 'script'));
  for (const filename of [
    'install.js',
    'install-utils.js',
    'install-metadata.js',
    'native-directory.js',
    'pack-platforms.js',
  ]) {
    fs.writeFileSync(path.join(source, 'script', filename), '// fixture');
  }
  fs.writeFileSync(path.join(source, 'LICENSE'), 'MIT fixture license');
  fs.writeFileSync(path.join(source, 'ThirdPartyNotices.txt'), 'fixture notices');
  for (const [platform, library] of Object.entries(libraries)) {
    for (const arch of ['x64', 'arm64']) {
      const directory = path.join(source, 'bin/napi-v6', platform, arch);
      fs.mkdirSync(directory, { recursive: true });
      fs.writeFileSync(path.join(directory, 'onnxruntime_binding.node'), `binding ${platform}/${arch}`);
      fs.writeFileSync(path.join(directory, library), `runtime ${platform}/${arch}`);
    }
  }
  if (process.platform !== 'win32') {
    const directory = path.join(source, 'bin/napi-v6/linux/x64');
    fs.renameSync(path.join(directory, libraries.linux), path.join(directory, 'libonnxruntime.so.1.32.0'));
    fs.symlinkSync('libonnxruntime.so.1.32.0', path.join(directory, libraries.linux));
  }
  archives = packPlatforms(source, path.join(root, 'archives'));
  unpacked = new Map();
  for (const archive of archives) {
    const destination = path.join(root, archive.name);
    fs.mkdirSync(destination);
    execFileSync('tar', ['-xzf', path.join(root, 'archives', archive.filename), '-C', destination]);
    unpacked.set(archive.name, path.join(destination, 'package'));
  }
});
after(() => fs.rmSync(root, { recursive: true, force: true }));

test('parent publishes only JavaScript and exact-version optional native dependencies', () => {
  const archive = archives.find((entry) => entry.name === 'onnxruntime-node');
  assert.equal(archives.length, 7);
  assert.equal(archives.at(-1), archive);
  assert.ok(archive.files.every((entry) => !entry.path.startsWith('bin/')));
  const parent = JSON.parse(fs.readFileSync(path.join(unpacked.get('onnxruntime-node'), 'package.json')));
  assert.equal(Object.keys(parent.optionalDependencies).length, 6);
  assert.ok(Object.values(parent.optionalDependencies).every((value) => value === version));
  assert.deepEqual(parent.scripts, { postinstall: 'node ./script/install.js' });
  for (const filename of ['install.js', 'install-utils.js', 'install-metadata.js', 'native-directory.js']) {
    assert.ok(archive.files.some((entry) => entry.path === `script/${filename}`));
  }
  assert.ok(!archive.files.some((entry) => entry.path === 'script/pack-platforms.js'));
  assert.equal(fs.readFileSync(path.join(source, 'package.json'), 'utf8'), originalManifest);
});

test('native packages preserve binary bytes and platform metadata, including SONAME links', () => {
  for (const archive of archives.filter((entry) => entry.name !== 'onnxruntime-node')) {
    const directory = unpacked.get(archive.name);
    const manifest = JSON.parse(fs.readFileSync(path.join(directory, 'package.json')));
    const [platform] = manifest.os;
    const [arch] = manifest.cpu;
    assert.equal(manifest.version, version);
    assert.equal(manifest.scripts, undefined);
    assert.deepEqual(manifest.libc, platform === 'linux' ? ['glibc'] : undefined);
    for (const filename of ['onnxruntime_binding.node', libraries[platform]]) {
      const relative = `bin/napi-v6/${platform}/${arch}/${filename}`;
      assert.ok(fs.existsSync(path.join(directory, relative)), JSON.stringify(archive.files));
      assert.deepEqual(fs.readFileSync(path.join(directory, relative)), fs.readFileSync(path.join(source, relative)));
      assert.ok(fs.lstatSync(path.join(directory, relative)).isFile());
    }
    assert.equal(fs.readFileSync(path.join(directory, 'LICENSE'), 'utf8'), 'MIT fixture license');
    assert.equal(fs.readFileSync(path.join(directory, 'ThirdPartyNotices.txt'), 'utf8'), 'fixture notices');
    assert.ok(
      archive.files.every(
        (entry) => !entry.path.startsWith('bin/') || entry.path.startsWith(`bin/napi-v6/${platform}/${arch}/`),
      ),
    );
  }
});

test('legacy and source builds continue to use their bundled binary', () => {
  assert.equal(
    getNativeBindingPath(source, 'linux', 'x64'),
    path.join(source, 'bin/napi-v6/linux/x64/onnxruntime_binding.node'),
  );
});

test('optional package resolution rejects missing and mismatched packages', () => {
  const parent = path.join(root, 'resolution');
  fs.mkdirSync(parent);
  writeJson(path.join(parent, 'package.json'), { name: 'onnxruntime-node', version });
  assert.throws(() => getNativeBindingPath(parent, 'linux', 'x64'), /Missing onnxruntime-node-linux-x64/);
  const native = path.join(parent, 'node_modules/onnxruntime-node-linux-x64');
  fs.cpSync(unpacked.get('onnxruntime-node-linux-x64'), native, { recursive: true });
  assert.equal(
    getNativeBindingPath(parent, 'linux', 'x64'),
    path.join(native, 'bin/napi-v6/linux/x64/onnxruntime_binding.node'),
  );
  writeJson(path.join(native, 'package.json'), { version: '1.31.0' });
  assert.throws(() => getNativeBindingPath(parent, 'linux', 'x64'), /Version mismatch/);
});

test('incomplete native payload fails before producing archives', () => {
  const incomplete = path.join(root, 'incomplete');
  fs.cpSync(source, incomplete, { recursive: true, dereference: true });
  fs.rmSync(path.join(incomplete, 'bin/napi-v6/linux/x64/onnxruntime_binding.node'));
  assert.throws(() => packPlatforms(incomplete, path.join(root, 'invalid-archives')), /Missing native payload/);
  assert.ok(!fs.existsSync(path.join(root, 'invalid-archives')));
});

test('npm installs only the host native package from the generated archives', () => {
  const consumer = path.join(root, 'consumer');
  fs.mkdirSync(consumer);
  const parentArchive = archives.find((entry) => entry.name === 'onnxruntime-node');
  const overrides = Object.fromEntries(
    archives
      .filter((entry) => entry !== parentArchive)
      .map((entry) => [entry.name, `file:${path.join(root, 'archives', entry.filename)}`]),
  );
  writeJson(path.join(consumer, 'package.json'), {
    name: 'consumer',
    version: '1.0.0',
    private: true,
    dependencies: { 'onnxruntime-node': `file:${path.join(root, 'archives', parentArchive.filename)}` },
    overrides,
  });
  execFileSync(
    process.execPath,
    [findNpmCli(), 'install', '--ignore-scripts', '--no-audit', '--no-fund', '--registry=http://127.0.0.1:9'],
    { cwd: consumer, stdio: 'pipe' },
  );
  const installed = fs
    .readdirSync(path.join(consumer, 'node_modules'))
    .filter((name) => name.startsWith('onnxruntime-node-'));
  assert.deepEqual(installed, [`onnxruntime-node-${process.platform}-${process.arch}`]);
  const parent = path.join(consumer, 'node_modules/onnxruntime-node');
  assert.ok(getNativeBindingPath(parent).includes(installed[0]));
});

test('installer skips optional downloads and puts requested libraries beside the selected binding', () => {
  const parent = path.join(root, 'installer');
  fs.mkdirSync(path.join(parent, 'script'), { recursive: true });
  writeJson(path.join(parent, 'package.json'), { name: 'onnxruntime-node', version });
  for (const filename of ['install.js', 'native-directory.js']) {
    fs.copyFileSync(path.join(__dirname, filename), path.join(parent, 'script', filename));
  }
  const agent = path.join(parent, 'node_modules/global-agent');
  fs.mkdirSync(agent, { recursive: true });
  fs.writeFileSync(path.join(agent, 'index.js'), 'exports.bootstrap = () => {};');
  const platform = `${process.platform}/${process.arch}`;
  fs.writeFileSync(
    path.join(parent, 'script/install-metadata.js'),
    `module.exports = ${JSON.stringify({
      requirements: { [platform]: ['addon'] },
      manifests: {
        [`${platform}:addon`]: { 'provider-library': { package: 'native-addon', path: 'runtime/provider-library' } },
      },
      packages: { 'native-addon': [{ id: 'fixture', version }] },
      feeds: [],
    })};`,
  );
  fs.writeFileSync(
    path.join(parent, 'script/install-utils.js'),
    `
    exports.parseInstallFlag = () => process.env.ONNXRUNTIME_NODE_INSTALL === 'skip' ? false : 'addon';
    exports.installPackages = (_packages, manifests) => process.stdout.write(JSON.stringify(manifests));
  `,
  );
  const install = (flag) =>
    execFileSync(process.execPath, [path.join(parent, 'script/install.js')], {
      env: { ...process.env, ONNXRUNTIME_NODE_INSTALL: flag },
      encoding: 'utf8',
    });
  assert.equal(install('skip'), '');
  const [sourceManifest] = JSON.parse(install('addon'));
  assert.equal(
    sourceManifest.filepath,
    path.join(parent, 'bin/napi-v6', process.platform, process.arch, 'provider-library'),
  );
  const nativeName = `onnxruntime-node-${process.platform}-${process.arch}`;
  writeJson(path.join(parent, 'package.json'), {
    name: 'onnxruntime-node',
    version,
    optionalDependencies: { [nativeName]: version },
  });
  const nativeRoot = path.join(parent, 'node_modules', nativeName);
  fs.cpSync(unpacked.get(nativeName), nativeRoot, { recursive: true });
  const [manifest] = JSON.parse(install('addon'));
  assert.equal(
    manifest.filepath,
    path.join(nativeRoot, 'bin/napi-v6', process.platform, process.arch, 'provider-library'),
  );
  assert.equal(manifest.pathInPackage, 'runtime/provider-library');
});
