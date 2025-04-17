// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

'use strict';

const path = require('path');
const { installOrtPackages, runShellCmd } = require('./utils');

/**
 * Entry point for type tests.
 *
 * @param {string[]} packagesToInstall
 */
module.exports = async function main(PRESERVE, PACKAGES_TO_INSTALL) {
  console.log('Running type tests...');

  // testcases/module-resolution
  {
    await installOrtPackages('module-resolution', PRESERVE, PACKAGES_TO_INSTALL);

    await runShellCmd('npx tsc', { wd: path.join(__dirname, 'testcases', 'module-resolution') });
  }
};
