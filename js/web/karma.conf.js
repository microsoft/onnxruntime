// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

'use strict';

const args = require('minimist')(process.argv, {});
const bundleMode = args['bundle-mode'] || 'dev';  // 'dev'|'perf'
const karmaPlugins = args['karma-plugins'] || undefined;
const timeoutMocha = args['timeout-mocha'] || 60000;
const forceLocalHost = !!args['force-localhost'];

// parse chromium flags
let chromiumFlags = args['chromium-flags'];
if (!chromiumFlags) {
  chromiumFlags = [];
} else if (typeof chromiumFlags === 'string') {
  chromiumFlags = [chromiumFlags];
} else if (!Array.isArray(chromiumFlags)) {
  throw new Error(`Invalid command line arg: --chromium-flags: ${chromiumFlags}`);
}

const ORT_FILE = bundleMode === 'dev' ? 'dist/ort.all.js' : 'dist/ort.all.min.js';
const TEST_FILE = bundleMode === 'dev' ? 'test/ort.test.js' : 'test/ort.test.min.js';

// it's a known issue that Safari does not work with "localhost" in BrowserStack:
// https://www.browserstack.com/question/663
//
// we need to read machine IP address to replace "localhost":
// https://stackoverflow.com/a/8440736
//
function getMachineIpAddress() {
  if (!forceLocalHost) {
    var os = require('os');
    var ifaces = os.networkInterfaces();

    for (const ifname in ifaces) {
      for (const iface of ifaces[ifname]) {
        if ('IPv4' !== iface.family || iface.internal !== false) {
          // skip over internal (i.e. 127.0.0.1) and non-ipv4 addresses
          continue;
        }

        // returns the first available IP address
        return iface.address;
      }
    }
  }

  // if no available IP address, fallback to "localhost".
  return 'localhost';
}

const hostname = getMachineIpAddress();
// In Node.js v16 and below, 'localhost' is using IPv4, so need to listen to '0.0.0.0'
// In Node.js v17+, 'localhost' is using IPv6, so need to listen to '::'
const listenAddress = Number.parseInt(process.versions.node.split('.')[0]) >= 17 ? '::' : '0.0.0.0';

module.exports = function(config) {
  config.set({
    // global config of your BrowserStack account
    browserStack: {
      username: process.env.BROWSERSTACK_USERNAME,
      accessKey: process.env.BROWSERSTACK_ACCESS_KEY,
      forceLocal: true,
      startTunnel: true,
      idleTimeout: '300',
    },
    frameworks: ['mocha'],
    files: [
      {pattern: ORT_FILE},
      {pattern: TEST_FILE},
      {pattern: 'test/testdata-file-cache-*.json', included: false, watched: false},
      {pattern: 'test/data/**/*', included: false, nocache: true, watched: false},
      {pattern: 'dist/*.wasm', included: false, watched: false},
    ],
    plugins: karmaPlugins,
    logLevel: config.LOG_DEBUG,
    client: {captureConsole: true, mocha: {expose: ['body'], timeout: timeoutMocha}},
    reporters: ['mocha', 'BrowserStack'],
    browsers: [],
    captureTimeout: 120000,
    reportSlowerThan: 100,
    browserDisconnectTimeout: 600000,
    browserNoActivityTimeout: 300000,
    browserDisconnectTolerance: 0,
    browserSocketTimeout: 60000,
    hostname,
    listenAddress,
    customLaunchers: {
      // the following flags are used to make sure Edge on CI agents to initialize WebGPU correctly.
      EdgeTest: {base: 'Edge', flags: chromiumFlags},
      ChromeTest: {base: 'Chrome', flags: chromiumFlags},
      ChromeTestHeadless: {base: 'ChromeHeadless', flags: chromiumFlags},
      ChromeCanaryTest: {base: 'ChromeCanary', flags: chromiumFlags},
      //
      // ==== BrowserStack browsers ====
      //

      // Windows
      //
      BS_WIN_10_Chrome_91:
          {base: 'BrowserStack', os: 'Windows', os_version: '10', browser: 'Chrome', browser_version: '91'},
      BS_WIN_10_Edge_91:
          {base: 'BrowserStack', os: 'Windows', os_version: '10', browser: 'Edge', browser_version: '91'},
      BS_WIN_10_Firefox_89:
          {base: 'BrowserStack', os: 'Windows', os_version: '10', browser: 'Firefox', browser_version: '89'},

      // macOS
      //
      BS_MAC_11_Safari_14:
          {base: 'BrowserStack', os: 'OS X', os_version: 'Big Sur', browser: 'Safari', browser_version: '14.0'},
      BS_MAC_11_Chrome_91:
          {base: 'BrowserStack', os: 'OS X', os_version: 'Big Sur', browser: 'Chrome', browser_version: '91'},

      // iPhone
      //
      BS_IOS_14_iPhoneXS: {base: 'BrowserStack', device: 'iPhone XS', real_mobile: true, os: 'ios', os_version: '14'},
      BS_IOS_13_iPhoneXS: {base: 'BrowserStack', device: 'iPhone XS', real_mobile: true, os: 'ios', os_version: '13'},

      // Android
      //
      BS_ANDROID_11_Pixel_5:
          {base: 'BrowserStack', device: 'Google Pixel 5', real_mobile: true, os: 'android', os_version: '11.0'},
      BS_ANDROID_11_Galaxy_S_21:
          {base: 'BrowserStack', device: 'Samsung Galaxy S21', real_mobile: true, os: 'android', os_version: '11.0'},
      BS_ANDROID_10_Pixel_4:
          {base: 'BrowserStack', device: 'Google Pixel 4', real_mobile: true, os: 'android', os_version: '10.0'}
    }
  });
};
