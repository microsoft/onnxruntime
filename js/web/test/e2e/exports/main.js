// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

'use strict';

const { runDevTest, runProdTest, verifyAssets } = require('./test');
const { installOrtPackages } = require('./utils');

/**
 * Entry point for package exports tests.
 *
 * @param {string[]} packagesToInstall
 */
module.exports = async function main(PRESERVE, PACKAGES_TO_INSTALL) {
  console.log('Running exports tests...');

  // testcases/nextjs-default
  {
    await installOrtPackages('nextjs-default', PRESERVE, PACKAGES_TO_INSTALL);

    await runDevTest('nextjs-default', '✓ Ready in', 3000);
    await runDevTest('nextjs-default', '✓ Ready in', 3000, 'turbopack', 'npm run dev -- --turbopack');
    await runProdTest('nextjs-default', '✓ Ready in', 3000);
  }

  // testcases/vite-default
  {
    await installOrtPackages('vite-default', PRESERVE, PACKAGES_TO_INSTALL);

    await runDevTest('vite-default', '\x1b[32m➜\x1b[39m  \x1b[1mLocal\x1b[22m:', 5173);
    await runProdTest('vite-default', '\x1b[32m➜\x1b[39m  \x1b[1mLocal\x1b[22m:', 4173);

    await verifyAssets('vite-default', async (cwd) => {
      const globby = await import('globby');

      return {
        test: 'File "dist/assets/**/ort.*.mjs" should not exist',
        success: globby.globbySync('dist/assets/**/ort.*.mjs', { cwd }).length === 0,
      };
    });
  }
};
