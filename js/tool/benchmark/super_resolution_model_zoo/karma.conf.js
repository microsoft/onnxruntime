// Karma configuration
const path = require('path')
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

module.exports = function(config) {
  config.set({
    basePath: './',
    frameworks: ['mocha'],
    files: [
      { pattern: 'dist/main.js' },
      { pattern: 'dist/onnx-wasm.wasm', included: false},
	    { pattern: 'dist/onnx-worker.js', included: false},
      { pattern: 'data/**/*', watched: false, included: false, served: true, nocache: true }
    ],
    proxies: {
      '/onnx-wasm.wasm': '/base/dist/onnx-wasm.wasm',
      '/onnx-worker.js': '/base/dist/onnx-worker.js',
	 },
    exclude: [
    ],
    // available preprocessors: https://npmjs.org/browse/keyword/karma-preprocessor
    preprocessors: {
    },
    reporters: ['mocha'],
    captureTimeout: 120000,
    reportSlowerThan: 100,
    browserDisconnectTimeout: 600000,
    browserNoActivityTimeout: 300000,
    browserDisconnectTolerance: 0,
    browserSocketTimeout: 60000,
    logLevel: config.LOG_VERBOSE,
    hostname: getMachineIpAddress(),
    customLaunchers: {
      ChromeTest: {base: 'Chrome', flags: ['--window-size=1,1']},
      ChromeDebug: {debug: true, base: 'Chrome', flags: ['--remote-debugging-port=9333']}
    },
    client: {
      captureConsole: true,
      mocha: {expose: ['body'], timeout: 3000000},
      browser: config.browsers,
      printMatches: false,
      // To enable pack, run 'PACK=1 npm run test'
      usePackedGlTexture: config.usePackedGlTexture==1 ? true : false,
      runIteration: config.runIteration ? config.runIteration : 10,
      profile: config.profile
    },
    browsers: ['ChromeTest', 'ChromeDebug', 'Edge', 'Safari'],
    browserConsoleLogOptions: {level: "debug", format: "%b %T: %m", terminal: true},
  })
}