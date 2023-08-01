// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {compareSync} from 'dir-compare';
import fs from 'fs-extra';
import jszip from 'jszip';
import log from 'npmlog';
import * as path from 'path';

import {downloadZip, extractFile} from './utils';

const TEST_DATA_OPSET_VERSIONS = [
  ['opset19', '1.14.0'],
  ['opset18', '1.13.1'],
  ['opset17', '1.12.1'],
  ['opset16', '1.11.0'],
  ['opset15', '1.10.2'],
  ['opset14', '1.9.1'],
  ['opset13', '1.8.1'],
  ['opset12', '1.7.0'],
  ['opset11', '1.6.1'],
  ['opset10', '1.5.0'],
  ['opset9', '1.4.1'],
  ['opset8', '1.3.0'],
  ['opset7', '1.2.3'],
];

const JS_ROOT = path.join(__dirname, '..');
const JS_TEST_ROOT = path.join(JS_ROOT, 'test');
const JS_TEST_DATA_ROOT = path.join(JS_TEST_ROOT, 'data');
const JS_TEST_DATA_NODE_ROOT = path.join(JS_TEST_DATA_ROOT, 'node');

const main = async () => {
  log.info('PrepareTestData', 'Preparing node tests ...');

  if (fs.existsSync(path.join(JS_TEST_DATA_NODE_ROOT, '__generated_onnx_node_tests'))) {
    return;
  }

  for (const opsetMapping of TEST_DATA_OPSET_VERSIONS) {
    const opset = opsetMapping[0];
    const onnxVersion = opsetMapping[1];

    const resourceUri = `https://github.com/onnx/onnx/archive/refs/heads/rel-${onnxVersion}.zip`;

    log.info('PrepareTestData', `Downloading onnx ${opset}(v${onnxVersion}): ${resourceUri}`);

    const folderPrefix = `onnx-rel-${onnxVersion}/onnx/backend/test/data/node`;

    const buffer = await downloadZip(resourceUri);
    const zip = await jszip.loadAsync(buffer);
    const entries = zip.filter(relativePath => relativePath.startsWith(folderPrefix));

    const testCasesFolder = path.join(JS_TEST_DATA_ROOT, 'node', opset);
    log.info('PrepareTestData', `Preparing folders under ${testCasesFolder}`);

    // create folders first
    for (const entry of entries) {
      if (entry.dir) {
        const folder = path.relative(folderPrefix, entry.name);
        if (folder) {
          await fs.ensureDir(path.join(testCasesFolder, folder));
        }
      }
    }

    // extract files
    log.info('PrepareTestData', `Extracting files to ${testCasesFolder}`);
    for (const entry of entries) {
      if (!entry.dir) {
        await extractFile(
            entry, fs.createWriteStream(path.join(testCasesFolder, path.relative(folderPrefix, entry.name))));
      }
    }
  }

  log.info('PrepareTestData', 'Deduplicating test cases...');
  for (let i = 0; i < TEST_DATA_OPSET_VERSIONS.length - 1; i++) {
    const currentOpset = TEST_DATA_OPSET_VERSIONS[i][0];
    const currentFolder = path.join(JS_TEST_DATA_NODE_ROOT, currentOpset);
    const previousOpset = TEST_DATA_OPSET_VERSIONS[i + 1][0];
    const previousFolder = path.join(JS_TEST_DATA_NODE_ROOT, previousOpset);

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
    log.info('PrepareTestData', `Deduplicated ${count} test case(s) in folder ${currentOpset}.`);
  }
  log.info('PrepareTestData', 'Deduplicating test cases... DONE');

  fs.ensureFileSync(path.join(JS_TEST_DATA_NODE_ROOT, '__generated_onnx_node_tests'));
};

void main();
