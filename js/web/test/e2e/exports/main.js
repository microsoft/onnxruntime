// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

'use strict';

const path = require('path');

/**
 * Entry point for package exports tests.
 *
 * @param {string[]} packagesToInstall
 */
module.exports = async function main() {
  console.log('Running exports tests...');

  // testcases/nextjs-default
  {
    const workingDir = path.join(__dirname, 'testcases/nextjs-default');
    await runShellCmd('npm install ../../onnxruntime-common.tgz ../../onnxruntime-web.tgz', workingDir);
  }
};

async function runShellCmd(cmd, wd = __dirname) {
  console.log('===============================================================');
  console.log(' Running command in shell:');
  console.log(' > ' + cmd);
  console.log('===============================================================');
  let complete = false;
  const childProcess = spawn(cmd, { shell: true, stdio: 'inherit', cwd: wd });
  childProcess.on('close', function (code) {
    if (code !== 0) {
      process.exit(code);
    } else {
      complete = true;
    }
  });
  while (!complete) {
    await delay(100);
  }
}
