// Host-side hook that lets a plugin EP supply its own JS glue at runtime.
//
// `wasmImports` is module-scope in the generated glue and cannot be exposed through
// EXPORTED_RUNTIME_METHODS directly, but a JS-library function defined here shares that scope.
// Exporting this one tiny function is enough: the host needs a *hook*, not the EP's glue.
//
// This is what onnxruntime.wasm would have to add so that an EP such as WebGPU could ship
// emdawnwebgpu's library_webgpu.js alongside the EP wasm instead of having it linked into the
// host (blocker #5).
addToLibrary({
  $ortRegisterEpJsGlue: (name, fn) => {
    // Must be called before the EP side module is instantiated: libdylink.js resolves the
    // module's `env` imports against wasmImports at load time.
    wasmImports[name] = fn;
  },
});
