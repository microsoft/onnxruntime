// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {spawnSync} from 'child_process';
import * as fs from 'fs-extra';
import minimist from 'minimist';
import npmlog from 'npmlog';
import * as path from 'path';

const ROOT = path.join(__dirname, '..', '..', '..');
const DEPS_ONNX = path.join(ROOT, 'cmake', 'external', 'onnx');
const TEST_DATA_ROOT = path.join(__dirname, '..', 'test', 'data');
const TEST_DATA_NODE = path.join(TEST_DATA_ROOT, 'node');

const args = minimist(process.argv);
// prepare test data only when eiter flag '-f' or '--force' is specified, or the folder does not exist.
if (args.f || args.force || !fs.existsSync(TEST_DATA_NODE)) {
  npmlog.info('PrepareTestData', 'Preparing node tests ...');
  fs.removeSync(TEST_DATA_NODE);
  [['v7', 'rel-1.2.3'],
   ['v8', 'rel-1.3.0'],
   ['v9', 'rel-1.4.1'],
   ['v10', 'rel-1.5.0'],
   ['v11', 'rel-1.6.1'],
   ['v12', 'rel-1.7.0'],
  ].forEach(v => {
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
}
