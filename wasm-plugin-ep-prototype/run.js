// Node driver for the wasm plugin-EP prototype.
//
// Usage: node run.js <build-dir> [preload|dlopen]
//
//   preload  : hand the side module to Emscripten at startup via Module.dynamicLibraries
//              (asynchronous compile -> the browser-safe pattern)
//   dlopen   : write the side module into MEMFS and let dlopen() compile it synchronously
//              (works in Node; hits Chrome's 4KB main-thread sync-compile limit in a browser)
//   ondemand : call dlopen() on a library that was never preloaded and is not in the FS.
//              Requires -sASYNCIFY; Emscripten fetches + compiles it asynchronously.

const fs = require('fs');
const path = require('path');

const buildDir = process.argv[2] || '.';
const mode = process.argv[3] || 'preload';

const hostPath = path.join(buildDir, 'ort_host.js');
const pluginWasm = path.join(buildDir, 'plugin_ep.wasm');

const createHost = require(path.resolve(hostPath));
const { registerEpJsGlue } = require('./ep_js_glue.js');

(async () => {
  const moduleArgs = {
    print: (t) => console.log(t),
    printErr: (t) => console.error(t),
    locateFile: (f) => path.join(path.resolve(buildDir), f),
    // Emscripten's MODULARIZE output uses the object passed to the factory AS `Module`, so this
    // closure sees the fully initialised module. preRun runs before dynamicLibraries are loaded,
    // which is what makes `preload` mode work -- the EP's glue must already be in wasmImports by
    // the time the side module is instantiated.
    preRun: [() => registerEpJsGlue(moduleArgs)],
  };

  if (mode === 'preload') {
    moduleArgs.dynamicLibraries = ['plugin_ep.wasm'];
  }

  const Module = await createHost(moduleArgs);

  if (mode === 'dlopen') {
    // Mirrors a browser doing: const bytes = await (await fetch(url)).arrayBuffer();
    Module.FS.writeFile('/plugin_ep.wasm', new Uint8Array(fs.readFileSync(pluginWasm)));
  }

  const libPath = mode === 'dlopen' ? '/plugin_ep.wasm' : 'plugin_ep.wasm';

  // With -sASYNCIFY (or -sJSPI) dlopen() suspends the calling export, so it returns a Promise.
  // That is the mechanism that makes on-demand EP loading viable in a browser, where
  // synchronous WebAssembly compilation over ~4MB is forbidden on the main thread.
  const callAsync = async (fn, argTypes, args) => {
    const r = Module.ccall(fn, 'number', argTypes, args, { async: true });
    return (r && typeof r.then === 'function') ? await r : r;
  };
  const call = (fn, argTypes, args) => Module.ccall(fn, 'number', argTypes, args);

  console.log(`\n=== register (mode=${mode}, path=${libPath}) ===`);
  let rc = await callAsync('OrtRegisterExecutionProviderLibrary', ['string', 'string'],
    ['WebGpuPrototype', libPath]);
  if (rc !== 0) {
    console.error(`FAILED: OrtRegisterExecutionProviderLibrary rc=${rc}`);
    process.exit(1);
  }

  console.log('\n=== error-path / OrtStatus round trip ===');
  rc = call('OrtTestErrorPath', [], []);
  if (rc !== 0) {
    console.error(`FAILED: OrtTestErrorPath rc=${rc}`);
    process.exit(1);
  }

  console.log('\n=== unregister ===');
  rc = call('OrtUnregisterExecutionProviderLibrary', [], []);
  if (rc !== 0) {
    console.error(`FAILED: OrtUnregisterExecutionProviderLibrary rc=${rc}`);
    process.exit(1);
  }

  console.log('\nPROTOTYPE PASSED');
})().catch((e) => {
  console.error('PROTOTYPE FAILED:', e);
  process.exit(1);
});
