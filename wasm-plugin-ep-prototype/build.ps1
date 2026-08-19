# Build the wasm plugin-EP prototype.
#
#   .\build.ps1 -MainModule 1        # whole-archive main module (proof of concept)
#   .\build.ps1 -MainModule 2        # dead-code-eliminated main module (size measurement)
#   .\build.ps1 -MainModule 1 -Pthreads
#   .\build.ps1 -MainModule 1 -Asyncify
#
# Assumes emsdk_env.bat has already been sourced by the caller (see run_all.ps1).

param(
    [ValidateSet('0', '1', '2')][string]$MainModule = '1',
    [switch]$Pthreads,
    [switch]$Asyncify,
    [switch]$Jspi,
    [switch]$WasmEH,
    [switch]$Pad,
    [switch]$EpJsGlue,
    [switch]$Assertions,
    [string]$OutDir = ''
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$ortInclude = Join-Path (Split-Path $root -Parent) 'include\onnxruntime\core\session'

if (-not $OutDir) {
    $OutDir = "build-mm$MainModule"
    if ($WasmEH) { $OutDir += '-wasmeh' }
    if ($Pthreads) { $OutDir += '-pthreads' }
    if ($Asyncify) { $OutDir += '-asyncify' }
    if ($Jspi) { $OutDir += '-jspi' }
    if ($Pad) { $OutDir += '-pad' }
    if ($EpJsGlue) { $OutDir += '-epglue' }
}
$out = Join-Path $root $OutDir
# Always start from a clean directory. Reusing it risks measuring or running a stale host/side
# module pair from a previous invocation, which can make a failed build look like a pass.
if (Test-Path $out) { Remove-Item -Recurse -Force $out }
New-Item -ItemType Directory -Force -Path $out | Out-Null

# NOTE: the exception model must match in BOTH modules. Emscripten links a different
# libc++/libc++abi variant depending on it, and a side module that throws while the main
# module was built without catching support aborts at the first throw.
#
# -fexceptions      = legacy JS-based EH. Needs invoke_* trampolines and the __THREW__ global,
#                     which only exist if the MAIN module itself generated them -> brittle
#                     under -sMAIN_MODULE=2.
# -fwasm-exceptions = native WebAssembly exception handling. No invoke_*/__THREW__ at all.
$ehFlag = if ($WasmEH) { '-fwasm-exceptions' } else { '-fexceptions' }
$common = @('-O2', '-fPIC', $ehFlag, '-I', $ortInclude, '-std=c++17')
$sideExtra = @()
$mainExtra = @()

if ($Pthreads) {
    $common += '-pthread'
    $mainExtra += @('-sPTHREAD_POOL_SIZE=2')
}
if ($Asyncify) {
    $mainExtra += @('-sASYNCIFY=1', '-sASYNCIFY_STACK_SIZE=65536')
    $sideExtra += @('-sASYNCIFY=1')
}
if ($Jspi) {
    # ORT-web's alternative to ASYNCIFY (onnxruntime_ENABLE_WEBASSEMBLY_JSPI). Unlike ASYNCIFY
    # it is compatible with -fwasm-exceptions and needs no whole-program instrumentation.
    $mainExtra += @('-sJSPI=1', '-sJSPI_EXPORTS=OrtRegisterExecutionProviderLibrary')
    $sideExtra += @('-sJSPI=1')
}

if ($Pad) { $sideExtra += '-DPROTOTYPE_PAD_SIDE_MODULE=1' }
if ($EpJsGlue) { $sideExtra += '-DPROTOTYPE_EP_JS_GLUE=1' }
# ASSERTIONS makes libdylink.js name the symbol in "undefined symbol '<name>'", which is how you
# find the next missing dependency when MAIN_MODULE=2 fails to resolve something.
if ($Assertions) { $mainExtra += '-sASSERTIONS=1' }

Write-Host "=== building side module (plugin EP) ===" -ForegroundColor Cyan
$sideArgs = $common + $sideExtra + @(
    '-sSIDE_MODULE=1',
    (Join-Path $root 'plugin_ep.cc'),
    '-o', (Join-Path $out 'plugin_ep.wasm')
)
& emcc @sideArgs
if ($LASTEXITCODE -ne 0) { throw "side module build failed" }

Write-Host "=== building main module (mini ORT host) ===" -ForegroundColor Cyan
# MAIN_MODULE=2 only keeps symbols that are explicitly exported, so the side module's
# imports must be listed by hand. This mirrors exactly the problem onnxruntime-web would
# face: the EP side module's undefined symbols become the host's required export list.
$exported = "_main,_malloc,_free,_OrtRegisterExecutionProviderLibrary,_OrtTestErrorPath,_OrtUnregisterExecutionProviderLibrary"
if ($MainModule -eq '2') {
    # Derive the required export list mechanically from the side module's `env` imports.
    # With MAIN_MODULE=2 the main module is dead-code-eliminated, so anything the plugin EP
    # imports must be named here or the load fails with "undefined symbol".
    $wasmDis = Join-Path (Split-Path $root -Parent) 'emsdk\upstream\bin\wasm-dis.exe'
    $dis = & $wasmDis (Join-Path $out 'plugin_ep.wasm') 2>$null
    # `env` imports are direct calls; GOT.func / GOT.mem imports are address-taken functions and
    # DATA symbols (C++ RTTI typeinfo, stderr, ...). All three kinds must be satisfied.
    $imports = $dis |
        Select-String -Pattern '\(import "(env|GOT\.func|GOT\.mem)"' |
        ForEach-Object { if ($_ -match '"(?:env|GOT\.func|GOT\.mem)"\s+"([^"]+)"') { $matches[1] } } |
        Sort-Object -Unique

    # Symbols that must NOT go in EXPORTED_FUNCTIONS because they are not wasm exports of the
    # main module:
    #   - linker/dylink-managed globals (__memory_base, __stack_pointer, memory, ...)
    #   - JS-library functions supplied through wasmImports (llvm_eh_typeid_for,
    #     __resumeException, emscripten_asm_const*, getTempRet0/setTempRet0)
    #   - symbols libdylink.js synthesises ON DEMAND in $resolveGlobalSymbol: invoke_* wrappers
    #     (via $createInvokeFunction) and __cxa_find_matching_catch_* variants. These genuinely
    #     do not need to pre-exist in the main module.
    #   - the side module's own EM_ASM sig-builder data
    #   - __asyncify_data / __asyncify_state: these globals are created by the ASYNCIFY Binaryen
    #     pass, which runs AFTER wasm-ld. Listing them in EXPORTED_FUNCTIONS fails at link time
    #     with "undefined exported symbol"; the pass exports them itself.
    #   - PrototypeEpJsGlue: EP-owned JS glue injected into `wasmImports` at runtime (models
    #     emdawnwebgpu's library_webgpu.js). The host has no such symbol to export.
    #
    # Everything else -- including wasm DATA symbols such as __THREW__ (legacy EH), C++ RTTI
    # typeinfo, and the __cpp_exception WebAssembly *tag* (native Wasm EH) -- IS a main-module
    # wasm export and must be listed.
    $jsProvided = '^(__indirect_function_table|__memory_base|__table_base|__stack_pointer|memory|' +
                  'invoke_.*|getTempRet0|setTempRet0|emscripten_asm_const.*|__cxa_find_matching_catch.*|' +
                  'llvm_eh_typeid_for|__resumeException|__asyncify_data|__asyncify_state|' +
                  'PrototypeEpJsGlue|_ZN20__em_asm_sig_builder.*)$'

    $needed = $imports | Where-Object { $_ -notmatch $jsProvided } | ForEach-Object { "_$_" }
    Write-Host "auto-derived side-module imports: $($needed -join ',')" -ForegroundColor Yellow
    $exported += "," + ($needed -join ',')

    # JS-library elements used ONLY by the dynamically loaded side module are dead-code-eliminated
    # out of the main module unless requested explicitly. Per the Emscripten dynamic linking docs
    # these must be pulled in with DEFAULT_LIBRARY_FUNCS_TO_INCLUDE.
    #
    # Use the plain (non-$) names. libcore.js defines the implementations as JS-only `$getTempRet0`
    # / `$setTempRet0` but also declares C-callable aliases `getTempRet0: '$getTempRet0'`. Only the
    # aliases land in `wasmImports`, and `wasmImports` is what libdylink.js resolves side-module
    # `env` imports against -- requesting the `$` form alone leaves the symbol undefined.
    $jsLibNeeded = $imports | Where-Object {
        $_ -in @('llvm_eh_typeid_for', '__resumeException')
    } | Sort-Object -Unique

    # getTempRet0/setTempRet0 cannot be satisfied by the stock library because libcore.js defines
    # them as JS-only ($-prefixed) symbols; the plain names are `__deps` aliases that never reach
    # `wasmImports`, which is the only table libdylink.js resolves side-module `env` imports
    # against. Supply them from a small custom JS library that redefines the plain names as
    # ordinary (C-callable) library functions -- and force their inclusion, since nothing in the
    # main module itself references them.
    $tempRet = $imports | Where-Object { $_ -in @('getTempRet0', 'setTempRet0') } | Sort-Object -Unique
    if ($tempRet) {
        Write-Host "side module needs $($tempRet -join ',') -> adding legacy-EH dylink shim" -ForegroundColor Yellow
        $mainExtra += '--js-library'
        $mainExtra += (Join-Path $root 'legacy_eh_dylink_shim.js')
        # Force emission: nothing in the main module references these, so they are otherwise
        # dead-stripped even when the shim redefines them.
        $exported += "," + (($tempRet | ForEach-Object { "_$_" }) -join ',')
    }

    if ($jsLibNeeded) {
        Write-Host "auto-derived JS-library deps: $($jsLibNeeded -join ',')" -ForegroundColor Yellow
        $mainExtra += "-sDEFAULT_LIBRARY_FUNCS_TO_INCLUDE=$($jsLibNeeded -join ',')"
    }
}

$runtimeMethods = "ccall,cwrap,FS,UTF8ToString,stringToUTF8"
# ortRegisterEpJsGlue is a host-side hook (host_ep_glue_hook.js) that lets a plugin EP install
# its own JS glue into wasmImports before dlopen, so the host does not have to link the EP's
# glue. See REPORT.md 3.6.
if ($MainModule -ne '0') {
    $runtimeMethods += ",loadDynamicLibrary,ortRegisterEpJsGlue"
    $mainExtra += '--js-library'
    $mainExtra += (Join-Path $root 'host_ep_glue_hook.js')
    # Required so addFunction() can materialise a function pointer for EP-provided JS glue that
    # the side module address-takes (see ep_js_glue.js).
    $mainExtra += '-sALLOW_TABLE_GROWTH=1'
}

$mainArgs = $common + $mainExtra + @(
    "-sMAIN_MODULE=$MainModule",
    '-sMODULARIZE=1',
    '-sEXPORT_ES6=0',
    '-sALLOW_MEMORY_GROWTH=1',
    '-sMAXIMUM_MEMORY=4294967296',
    '-sWASM_BIGINT=1',
    '-sEXIT_RUNTIME=0',
    '-sFORCE_FILESYSTEM=1',
    "-sEXPORTED_FUNCTIONS=$exported",
    "-sEXPORTED_RUNTIME_METHODS=$runtimeMethods",
    (Join-Path $root 'host_main.cc'),
    '-o', (Join-Path $out 'ort_host.js')
)
& emcc @mainArgs
if ($LASTEXITCODE -ne 0) { throw "main module build failed" }

Write-Host "=== artifacts ===" -ForegroundColor Cyan
Get-ChildItem $out -File | Where-Object { $_.Extension -in '.js', '.wasm' } |
    Select-Object Name, @{n = 'KB'; e = { [math]::Round($_.Length / 1KB, 1) } } | Format-Table -AutoSize
