// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

'use strict';

const THREADS = process.argv.includes('--wasm-threads');

const files = [];
const proxies = {};
const chromeFlags = ['--no-sandbox'];
for (const arg of process.argv) {
  if (arg.startsWith('--entry=')) {
    const entry = arg.substring('--entry='.length);
    files.push({pattern: `${entry}.js`, watched: false});
    files.push({pattern: `${entry}.data`, included: false, watched: false, nocache: true });
    files.push({pattern: `${entry}.wasm`, included: false});
    proxies[`/${entry}.data`] = `/base/${entry}.data`;
    if (THREADS) {
      files.push({pattern: `${entry}.worker.js`, included: false, watched: false });
      proxies[`/${entry}.worker.js`] = `/base/${entry}.worker.js`;
      chromeFlags.push('--enable-features=SharedArrayBuffer');
    }
    break;
  }
}

if (files.length === 0) {
  console.error('No entry file specified. Use --entry= to specify the entry file.');
  process.exit(1);
}

// gtest reporter writes the test results to the file specified by the --gtest_output flag.
const gtestReporter = {'reporter:gtest': ['type', function() {
  this.onBrowserComplete = function(browser, result) {
    if (result.file) {
      require('fs').writeFileSync(result.file, result.data);
    }
  };
}]};

module.exports = function(config) {
    config.set({
      basePath: '.',
      files,
      proxies,
      mime: {
        'application/octet-stream': ['data']
      },
      plugins: [
        require('karma-chrome-launcher'),
        require('@chiragrupani/karma-chromium-edge-launcher'),
        gtestReporter],
      browsers: ['ChromeTest'],
      reporters: ['progress', 'gtest'],
      client: {
        captureConsole: true,
        // Pass the gtest flags to the test runner
        args: process.argv.filter(arg => arg.startsWith('--gtest_'))
      },
      browserDisconnectTimeout: 600000,
      // allow running tests for 30 minutes
      browserNoActivityTimeout: 30 * 60 * 1000,
      customLaunchers: {
        ChromeTest: {
          base: 'ChromeCanary',
          flags: chromeFlags
        },
        ChromeHeadlessTest: {
          base: 'ChromeHeadless',
          flags: chromeFlags
        },
        EdgeTest: {
          base: 'Edge',
          flags: chromeFlags
        }
      }
    });
  };
