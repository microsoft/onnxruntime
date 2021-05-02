// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

const bundleMode = require('minimist')(process.argv)['bundle-mode'] || 'dev';  // 'dev'|'perf'|undefined;
const karmaPlugins = require('minimist')(process.argv)['karma-plugins'] || undefined;
const commonFile = bundleMode === 'dev' ? '../common/dist/ort-common.js' : '../common/dist/ort-common.min.js'
const mainFile = bundleMode === 'dev' ? 'test/ort.dev.js' : 'test/ort.perf.js';

// it's a known issue that Safari does not work with "localhost" in BrowserStack:
// https://www.browserstack.com/question/663
//
// we need to read machine IP address to replace "localhost":
// https://stackoverflow.com/a/8440736
//
function getMachineIpAddress() {
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

  // if no available IP address, fallback to "localhost".
  return 'localhost';
}

module.exports = function (config) {
  config.set({
    // global config of your BrowserStack account
    browserStack: {
      username: process.env.BROWSER_STACK_USERNAME,
      accessKey: process.env.BROWSER_STACK_ACCESS_KEY,
      forceLocal: true,
      startTunnel: true,
    },
    frameworks: ['mocha'],
    files: [
      { pattern: commonFile },
      { pattern: 'test/testdata-config.js' },
      { pattern: mainFile },
      { pattern: 'test/testdata-file-cache-*.json', included: false },
      //{ pattern: 'test/onnx-worker.js', included: false },
      { pattern: 'test/data/**/*', included: false, nocache: true },
      { pattern: 'dist/onnxruntime_wasm.wasm', included: false },
      { pattern: 'dist/onnxruntime_wasm_threads.wasm', included: false },
      { pattern: 'dist/onnxruntime_wasm_threads.worker.js', included: false },
    ],
    proxies: {
      '/base/test/onnxruntime_wasm.wasm': '/base/dist/onnxruntime_wasm.wasm',
      '/onnxruntime_wasm_threads.wasm': '/base/dist/onnxruntime_wasm_threads.wasm',
      '/onnxruntime_wasm_threads.worker.js': '/base/dist/onnxruntime_wasm_threads.worker.js',
    },
    plugins: karmaPlugins,
    client: { captureConsole: true, mocha: { expose: ['body'], timeout: 60000 } },
    preprocessors: { mainFile: ['sourcemap'] },
    reporters: ['mocha', 'BrowserStack'],
    browsers: [],
    captureTimeout: 120000,
    reportSlowerThan: 100,
    browserDisconnectTimeout: 600000,
    browserNoActivityTimeout: 300000,
    browserDisconnectTolerance: 0,
    browserSocketTimeout: 60000,
    hostname: getMachineIpAddress(),
    customLaunchers: {
      ChromeTest: { base: 'Chrome', flags: ['--window-size=1,1'] },
      ChromeDebug: { debug: true, base: 'Chrome', flags: ['--remote-debugging-port=9333'] },
      //
      // ==== BrowserStack browsers ====
      //

      // Windows
      //
      BS_WIN_10_Chrome_73: {
        base: 'BrowserStack',
        browser: 'Chrome',
        browser_version: '73.0',
        os: 'Windows',
        os_version: '10',
      },
      BS_WIN_10_Edge_18: {
        base: 'BrowserStack',
        os: 'Windows',
        os_version: '10',
        browser: 'Edge',
        browser_version: '18.0',
      },
      BS_WIN_10_Firefox_66: {
        base: 'BrowserStack',
        os: 'Windows',
        os_version: '10',
        browser: 'Firefox',
        browser_version: '66.0',
      },
      BS_WIN_7_Chrome_63: {
        base: 'BrowserStack',
        browser: 'Chrome',
        browser_version: '63.0',
        os: 'Windows',
        os_version: '7',
      },

      // macOS
      //
      BS_MAC_10_14_Safari_12: {
        base: 'BrowserStack',
        os: 'OS X',
        os_version: 'Mojave',
        browser: 'Safari',
        browser_version: '12.0',
      },
      BS_MAC_10_14_Chrome_73: {
        base: 'BrowserStack',
        os: 'OS X',
        os_version: 'Mojave',
        browser: 'Chrome',
        browser_version: '73.0',
      },
      BS_MAC_10_13_Safari_11_1: {
        base: 'BrowserStack',
        os: 'OS X',
        os_version: 'High Sierra',
        browser: 'Safari',
        browser_version: '11.1',
      },

      // iPhone
      //
      BS_IOS_12_1_iPhoneXS: {
        base: 'BrowserStack',
        device: 'iPhone XS',
        real_mobile: true,
        os: 'ios',
        os_version: '12.1',
      },
      BS_IOS_11_iPhoneX: {
        base: 'BrowserStack',
        device: 'iPhone X',
        real_mobile: true,
        os: 'ios',
        os_version: '11',
      },
      BS_IOS_10_3_iPhone7: {
        base: 'BrowserStack',
        device: 'iPhone 7',
        real_mobile: true,
        os: 'ios',
        os_version: '10.3',
      },

      // Android
      //
      BS_ANDROID_9_Pixel_3: {
        base: 'BrowserStack',
        device: 'Google Pixel 3',
        real_mobile: true,
        os: 'android',
        os_version: '9.0',
      },
      BS_ANDROID_7_1_Galaxy_Note_8: {
        base: 'BrowserStack',
        device: 'Samsung Galaxy Note 8',
        real_mobile: true,
        os: 'android',
        os_version: '7.1',
      },
    }
  });
};
