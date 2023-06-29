// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

'use strict';

const bundleMode = require('minimist')(process.argv)['bundle-mode'] || 'dev';  // 'dev'|'perf'|undefined;
const karmaPlugins = require('minimist')(process.argv)['karma-plugins'] || undefined;
const timeoutMocha = require('minimist')(process.argv)['timeout-mocha'] || 60000;
const forceLocalHost = !!require('minimist')(process.argv)['force-localhost'];
const commonFile = bundleMode === 'dev' ? '../common/dist/ort-common.js' : '../common/dist/ort-common.min.js'
const mainFile = bundleMode === 'dev' ? 'test/ort.dev.js' : 'test/ort.perf.js';

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

module.exports = function (config) {
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
      { pattern: commonFile },
      { pattern: mainFile },
      { pattern: 'test/testdata-file-cache-*.json', included: false },
      { pattern: 'test/data/**/*', included: false, nocache: true },
      { pattern: 'dist/ort-wasm.wasm', included: false },
      { pattern: 'dist/ort-wasm-threaded.wasm', included: false },
      { pattern: 'dist/ort-wasm-simd.wasm', included: false },
      { pattern: 'dist/ort-wasm-simd-threaded.wasm', included: false },
      { pattern: 'dist/ort-wasm-simd.jsep.wasm', included: false },
      { pattern: 'dist/ort-wasm-simd-threaded.jsep.wasm', included: false },
      { pattern: 'dist/ort-wasm-threaded.worker.js', included: false },
    ],
    proxies: {
      '/base/test/ort-wasm.wasm': '/base/dist/ort-wasm.wasm',
      '/base/test/ort-wasm-threaded.wasm': '/base/dist/ort-wasm-threaded.wasm',
      '/base/test/ort-wasm-simd.wasm': '/base/dist/ort-wasm-simd.wasm',
      '/base/test/ort-wasm-simd-threaded.wasm': '/base/dist/ort-wasm-simd-threaded.wasm',
      '/base/test/ort-wasm-simd.jsep.wasm': '/base/dist/ort-wasm-simd.jsep.wasm',
      '/base/test/ort-wasm-simd-threaded.jsep.wasm': '/base/dist/ort-wasm-simd-threaded.jsep.wasm',
      '/base/test/ort-wasm-threaded.worker.js': '/base/dist/ort-wasm-threaded.worker.js',
    },
    plugins: karmaPlugins,
    client: { captureConsole: true, mocha: { expose: ['body'], timeout: timeoutMocha } },
    preprocessors: { mainFile: ['sourcemap'] },
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
      ChromeTest: {
        base: 'Chrome',
        flags: ['--enable-features=SharedArrayBuffer']
      },
      ChromeTestHeadless: {
        base: 'ChromeHeadless',
        flags: ['--enable-features=SharedArrayBuffer']
      },
      ChromeDebug: {
        debug: true,
        base: 'Chrome', flags: ['--remote-debugging-port=9333', '--enable-features=SharedArrayBuffer']
      },
      ChromeCanaryTest: {
        base: 'ChromeCanary',
        flags: [
          '--enable-features=SharedArrayBuffer',
          '--enable-experimental-web-platform-features'
        ]
      },
      ChromeCanaryDebug: {
        debug: true,
        base: 'ChromeCanary',
        flags: [
          '--remote-debugging-port=9333',
          '--enable-features=SharedArrayBuffer',
          '--enable-experimental-web-platform-features'
        ]
      },
      ChromeWebGpuProfileTest: {
        base: 'Chrome',
        flags: [
          '--window-size=1,1',
          '--enable-features=SharedArrayBuffer',
          '--disable-dawn-features=disallow_unsafe_apis'
        ]
      },
      ChromeWebGpuProfileDebug: {
        debug: true,
        base: 'Chrome',
        flags: [
          '--remote-debugging-port=9333',
          '--enable-features=SharedArrayBuffer',
          '--disable-dawn-features=disallow_unsafe_apis',
        ]
      },
      //
      // ==== BrowserStack browsers ====
      //

      // Windows
      //
      BS_WIN_10_Chrome_91: {
        base: 'BrowserStack',
        os: 'Windows',
        os_version: '10',
        browser: 'Chrome',
        browser_version: '91'
      },
      BS_WIN_10_Edge_91: {
        base: 'BrowserStack',
        os: 'Windows',
        os_version: '10',
        browser: 'Edge',
        browser_version: '91'
      },
      BS_WIN_10_Firefox_89: {
        base: 'BrowserStack',
        os: 'Windows',
        os_version: '10',
        browser: 'Firefox',
        browser_version: '89'
      },

      // macOS
      //
      BS_MAC_11_Safari_14: {
        base: 'BrowserStack',
        os: 'OS X',
        os_version: 'Big Sur',
        browser: 'Safari',
        browser_version: '14.0'
      },
      BS_MAC_11_Chrome_91: {
        base: 'BrowserStack',
        os: 'OS X',
        os_version: 'Big Sur',
        browser: 'Chrome',
        browser_version: '91'
      },

      // iPhone
      //
      BS_IOS_14_iPhoneXS: {
        base: 'BrowserStack',
        device: 'iPhone XS',
        real_mobile: true,
        os: 'ios',
        os_version: '14'
      },
      BS_IOS_13_iPhoneXS: {
        base: 'BrowserStack',
        device: 'iPhone XS',
        real_mobile: true,
        os: 'ios',
        os_version: '13'
      },

      // Android
      //
      BS_ANDROID_11_Pixel_5: {
        base: 'BrowserStack',
        device: 'Google Pixel 5',
        real_mobile: true,
        os: 'android',
        os_version: '11.0'
      },
      BS_ANDROID_11_Galaxy_S_21: {
        base: 'BrowserStack',
        device: 'Samsung Galaxy S21',
        real_mobile: true,
        os: 'android',
        os_version: '11.0'
      },
      BS_ANDROID_10_Pixel_4: {
        base: 'BrowserStack',
        device: 'Google Pixel 4',
        real_mobile: true,
        os: 'android',
        os_version: '10.0'
      }
    }
  });
};
