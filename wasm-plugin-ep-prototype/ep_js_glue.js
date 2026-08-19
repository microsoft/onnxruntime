// EP-owned JS glue, shipped ALONGSIDE the plugin EP side module rather than linked into the host.
//
// This models emdawnwebgpu's `library_webgpu.js`. Today that glue is pulled into the final
// executable because `emdawnwebgpu_cpp` is linked PUBLIC into onnxruntime_providers_webgpu, and
// an Emscripten side module cannot contribute `--js-library` glue to the main module at link
// time. That is the basis of blocker #5.
//
// libdylink.js resolves a side module's `env` imports against `wasmImports` at dlopen time
// ($resolveGlobalSymbol -> isSymbolDefined -> wasmImports[symName]). So glue registered through
// the host's `ortRegisterEpJsGlue` hook BEFORE dlopen is picked up, and the host itself never
// needs to contain the glue -- only the one-line hook.
//
// Usage: registerEpJsGlue(Module) before calling OrtRegisterExecutionProviderLibrary.

function registerEpJsGlue(Module) {
  if (!Module.ortRegisterEpJsGlue) {
    throw new Error('host does not expose ortRegisterEpJsGlue (see host_ep_glue_hook.js)');
  }
  // A real EP would install its whole JS library here (WebGPU device/queue/buffer shims, ...).
  const glue = (value) => {
    console.log(`[ep-glue] PrototypeEpJsGlue(${value}) called from the side module`);
    return value + 22;
  };
  // Emscripten needs a signature whenever the symbol is address-taken rather than called
  // directly, because it has to materialise a real function pointer via addFunction(). Legacy
  // JS-based EH does exactly that: the call is routed through an invoke_* trampoline, so the
  // import becomes a GOT.func entry resolved eagerly at dlopen time. Without `.sig` that fails
  // with "Cannot read properties of undefined (reading 'slice')". 'ii' = i32 return, i32 arg.
  glue.sig = 'ii';

  Module.ortRegisterEpJsGlue('PrototypeEpJsGlue', glue);
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = { registerEpJsGlue };
}
