// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {spawnSync} from 'child_process';
import {compareSync} from 'dir-compare';
import * as fs from 'fs-extra';
import minimist from 'minimist';
import npmlog from 'npmlog';
import * as path from 'path';

const ROOT = path.join(__dirname, '..', '..', '..');
const DEPS_ONNX = path.join(ROOT, 'cmake', 'external', 'onnx');
const TEST_DATA_ROOT = path.join(__dirname, '..', 'test', 'data');
const TEST_DATA_NODE = path.join(TEST_DATA_ROOT, 'node');

const TEST_DATA_OPSET_VERSIONS = [
  ['v12', 'rel-1.7.0'],
  ['v11', 'rel-1.6.1'],
  ['v10', 'rel-1.5.0'],
  ['v9', 'rel-1.4.1'],
  ['v8', 'rel-1.3.0'],
  ['v7', 'rel-1.2.3'],
];

const args = minimist(process.argv);
// prepare test data only when eiter flag '-f' or '--force' is specified, or the folder does not exist.
if (args.f || args.force || !fs.existsSync(TEST_DATA_NODE)) {
  npmlog.info('PrepareTestData', 'Preparing node tests ...');
  fs.removeSync(TEST_DATA_NODE);
  TEST_DATA_OPSET_VERSIONS.forEach(v => {
    const version = v[0];
    const commit = v[1];
    npmlog.info('PrepareTestData', `Checking out deps/onnx ${commit}...`);
    const checkout = spawnSync(`git checkout -q -f ${commit}`, {shell: true, stdio: 'inherit', cwd: DEPS_ONNX});
    if (checkout.status !== 0) {
      if (checkout.error) {
        console.error(checkout.error);
      }
      process.exit(checkout.status === null ? undefined : checkout.status);
    }
    const from = path.join(DEPS_ONNX, 'onnx/backend/test/data/node');
    const to = path.join(TEST_DATA_NODE, version);
    npmlog.info('PrepareTestData', `Copying folders from "${from}" to "${to}"...`);
    fs.copySync(from, to);
  });

  npmlog.info('PrepareTestData', 'Revert git index...');
  const update = spawnSync(`git submodule update ${DEPS_ONNX}`, {shell: true, stdio: 'inherit', cwd: ROOT});
  if (update.status !== 0) {
    if (update.error) {
      console.error(update.error);
    }
    process.exit(update.status === null ? undefined : update.status);
  }
  npmlog.info('PrepareTestData', 'Revert git index... DONE');

  npmlog.info('PrepareTestData', 'Deduplicating test cases...');
  for (let i = 0; i < TEST_DATA_OPSET_VERSIONS.length - 1; i++) {
    const currentVersion = TEST_DATA_OPSET_VERSIONS[i][0];
    const currentFolder = path.join(TEST_DATA_NODE, currentVersion);
    const previousVersion = TEST_DATA_OPSET_VERSIONS[i + 1][0];
    const previousFolder = path.join(TEST_DATA_NODE, previousVersion);

    // compare each subfolder to its previous version. If they are same, remove the one in current version.
    let count = 0;
    fs.readdirSync(currentFolder, {withFileTypes: true}).forEach(dir => {
      const currentDir = path.join(currentFolder, dir.name);
      const previousDir = path.join(previousFolder, dir.name);
      if (dir.isDirectory() && fs.existsSync(previousDir) && fs.statSync(previousDir).isDirectory()) {
        if (compareSync(currentDir, previousDir, {compareContent: true}).differences === 0) {
          fs.removeSync(currentDir);
          count++;
        }
      }
    });
    npmlog.info('PrepareTestData', `Deduplicated ${count} test case(s) in folder ${currentVersion}.`);
  }
  npmlog.info('PrepareTestData', 'Deduplicating test cases... DONE');
}
