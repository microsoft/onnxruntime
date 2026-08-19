// Supplies the legacy-EH helpers that a dynamically loaded side module imports from `env`.
//
// Background: with -fexceptions (legacy JS-based EH) the compiler lowers `catch` into a call to
// __cxa_find_matching_catch_N, whose second return value is read back via `getTempRet0`. A side
// module therefore imports `env.getTempRet0`.
//
// libdylink.js resolves side-module `env` imports against `wasmImports` only. In libcore.js
// getTempRet0/setTempRet0 are defined as JS-only symbols ($getTempRet0/$setTempRet0); the plain
// names exist purely as aliases for `__deps` entries and never reach `wasmImports`. So with
// -sMAIN_MODULE=2 there is no stock way to satisfy the import, even with
// DEFAULT_LIBRARY_FUNCS_TO_INCLUDE.
//
// Defining them here as ordinary (non-$) library functions puts them in `wasmImports`, which is
// what the side module needs. -sMAIN_MODULE=1 does not need this shim.
addToLibrary({
  getTempRet0__deps: ['_emscripten_tempret_get'],
  getTempRet0: () => __emscripten_tempret_get(),

  setTempRet0__deps: ['_emscripten_tempret_set'],
  setTempRet0: (val) => __emscripten_tempret_set(val),
});
