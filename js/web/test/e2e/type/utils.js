// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

'use strict';

const path = require('path');
const { spawn } = require('child_process');
const treeKill = require('tree-kill');

async function installOrtPackages(testCaseName, PRESERVE, PACKAGES_TO_INSTALL) {
  if (!PRESERVE) {
    const wd = path.join(__dirname, 'testcases', testCaseName);
    if (PACKAGES_TO_INSTALL.length === 0) {
      await runShellCmd('npm ci', { wd });
    } else {
      await runShellCmd(`npm install ${PACKAGES_TO_INSTALL.map((i) => `"${i}"`).join(' ')}`, { wd });
    }
  }
}

async function runShellCmd(cmd, { wd = __dirname, event = null, ready = null, ignoreExitCode = false }) {
  console.log('===============================================================');
  console.log(' Running command in shell:');
  console.log(' > ' + cmd);
  console.log('===============================================================');

  return new Promise((resolve, reject) => {
    const childProcess = spawn(cmd, { shell: true, stdio: ['ignore', 'pipe', 'inherit'], cwd: wd });
    childProcess.on('close', function (code, signal) {
      if (code === 0 || ignoreExitCode) {
        resolve();
      } else {
        reject(`Process exits with code ${code}`);
      }
    });
    childProcess.stdout.on('data', (data) => {
      process.stdout.write(data);

      if (ready && event && data.toString().includes(ready)) {
        event.emit('serverReady');
      }
    });
    if (event) {
      event.on('kill', () => {
        childProcess.stdout.destroy();
        treeKill(childProcess.pid);
        console.log('killing process...');
      });
    }
  });
}

module.exports = {
  runShellCmd,
  installOrtPackages,
};
