// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

'use strict';

const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { execFileSync } = require('node:child_process');

const nativeLibraries = { linux: 'libonnxruntime.so.1', darwin: 'libonnxruntime.1.dylib', win32: 'onnxruntime.dll' };
const copyNativePayload = (source, destination) => {
  fs.mkdirSync(destination, { recursive: true });
  for (const filename of fs.readdirSync(source)) {
    const input = path.join(source, filename);
    const output = path.join(destination, filename);
    if (fs.statSync(input).isDirectory()) {
      copyNativePayload(input, output);
    } else {
      fs.copyFileSync(input, output);
    }
  }
};

const readJson = (filename) => JSON.parse(fs.readFileSync(filename, 'utf8'));
const writeJson = (filename, value) => fs.writeFileSync(filename, `${JSON.stringify(value, null, 2)}\n`);

// Calling npm through Node also works on Windows, without shell-quoting npm.cmd.
const findNpmCli = () => {
  const candidates = [
    process.env.npm_execpath,
    path.join(path.dirname(process.execPath), 'node_modules/npm/bin/npm-cli.js'),
    path.join(path.dirname(process.execPath), '../lib/node_modules/npm/bin/npm-cli.js'),
  ];
  const cli = candidates.find(
    (candidate) => candidate && path.basename(candidate) === 'npm-cli.js' && fs.existsSync(candidate),
  );
  if (!cli) {
    throw new Error('Cannot find npm-cli.js. Run this script with npm installed alongside Node.js.');
  }
  return cli;
};

const packPlatforms = (packageRoot = path.join(__dirname, '..'), destination = packageRoot) => {
  packageRoot = path.resolve(packageRoot);
  destination = path.resolve(destination);
  const parent = readJson(path.join(packageRoot, 'package.json'));
  if (parent.name !== 'onnxruntime-node' || !parent.version) {
    throw new Error('Expected an onnxruntime-node package with a version.');
  }
  if (Object.values(parent.dependencies || {}).some((version) => version.startsWith('file:'))) {
    throw new Error('Run npm run prepack before packing platform packages.');
  }

  const platforms = [];
  const binaryRoot = path.join(packageRoot, 'bin/napi-v6');
  for (const platform of fs.readdirSync(binaryRoot).sort()) {
    if (!nativeLibraries[platform]) {
      throw new Error(`Unsupported native platform: ${platform}`);
    }
    for (const arch of fs.readdirSync(path.join(binaryRoot, platform)).sort()) {
      if (arch !== 'x64' && arch !== 'arm64') {
        throw new Error(`Unsupported native architecture: ${arch}`);
      }
      const relativeDirectory = `bin/napi-v6/${platform}/${arch}`;
      const source = path.join(packageRoot, relativeDirectory);
      for (const filename of ['onnxruntime_binding.node', nativeLibraries[platform]]) {
        if (!fs.existsSync(path.join(source, filename))) {
          throw new Error(`Missing native payload: ${relativeDirectory}/${filename}`);
        }
      }
      platforms.push({ platform, arch, relativeDirectory, source, name: `onnxruntime-node-${platform}-${arch}` });
    }
  }
  if (platforms.length === 0) {
    throw new Error('No native platforms found.');
  }

  const npmCli = findNpmCli();
  const npm = (args, cwd) =>
    JSON.parse(
      execFileSync(process.execPath, [npmCli, ...args, '--json', '--ignore-scripts'], { cwd, encoding: 'utf8' }),
    );
  const staging = fs.mkdtempSync(path.join(os.tmpdir(), 'ort-node-platforms-'));
  try {
    const parentRoot = path.join(staging, 'parent');
    // Pack a staging copy so npm lifecycle scripts cannot rebuild or modify the source payload.
    // Keep .npmignore in the copy: npm remains responsible for the final publish file list.
    const excludedDirectories = new Set(['bin', 'node_modules', 'build', 'src', 'test', '.git', '.vscode']);
    fs.cpSync(packageRoot, parentRoot, {
      recursive: true,
      dereference: true,
      filter: (filename) => {
        const relative = path.relative(packageRoot, filename);
        return !excludedDirectories.has(relative.split(path.sep)[0]) && !relative.endsWith('.tgz');
      },
    });
    parent.optionalDependencies = { ...parent.optionalDependencies };
    for (const { name } of platforms) {
      parent.optionalDependencies[name] = parent.version;
    }
    // Development lifecycle scripts need TypeScript and must not run in the staged package.
    parent.scripts = parent.scripts && parent.scripts.postinstall ? { postinstall: parent.scripts.postinstall } : {};
    writeJson(path.join(parentRoot, 'package.json'), parent);

    const nativeRoots = [];
    for (const { name, platform, arch, relativeDirectory, source } of platforms) {
      const nativeRoot = path.join(staging, name);
      fs.mkdirSync(path.join(nativeRoot, path.dirname(relativeDirectory)), { recursive: true });
      // npm omits symbolic links from tarballs, including native library SONAME links.
      copyNativePayload(source, path.join(nativeRoot, relativeDirectory));
      writeJson(path.join(nativeRoot, 'package.json'), {
        name,
        version: parent.version,
        description: `ONNX Runtime Node.js native binaries for ${platform} ${arch}`,
        license: parent.license,
        repository: parent.repository,
        os: [platform],
        cpu: [arch],
        ...(platform === 'linux' ? { libc: ['glibc'] } : {}),
        main: `${relativeDirectory}/onnxruntime_binding.node`,
        files: [relativeDirectory, 'LICENSE', 'ThirdPartyNotices.txt'],
      });
      for (const filename of ['LICENSE', 'ThirdPartyNotices.txt']) {
        const candidates = [path.join(packageRoot, filename), path.join(packageRoot, '../..', filename)];
        const license = candidates.find((candidate) => fs.existsSync(candidate));
        if (license) {
          fs.copyFileSync(license, path.join(nativeRoot, filename));
        }
      }
      nativeRoots.push(nativeRoot);
    }
    fs.mkdirSync(destination, { recursive: true });
    // Release automation must publish these dependencies before the parent package.
    return [...nativeRoots, parentRoot].flatMap((cwd) => npm(['pack', '--pack-destination', destination], cwd));
  } finally {
    fs.rmSync(staging, { recursive: true, force: true });
  }
};

module.exports = { packPlatforms, findNpmCli };
if (require.main === module) {
  process.stdout.write(`${JSON.stringify(packPlatforms(), null, 2)}\n`);
}
