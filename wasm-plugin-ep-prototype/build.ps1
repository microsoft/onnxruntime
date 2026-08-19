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
}
$out = Join-Path $root $OutDir
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

    # These are provided by the main module's JS glue (library_*.js), not by wasm exports:
    # invoke_* trampolines, the Emscripten EH helpers, EM_ASM support, and the linker-managed
    # dylink globals. They must not be listed in EXPORTED_FUNCTIONS.
    # NOTE: __cpp_exception is a WebAssembly *tag* (native Wasm EH). It is NOT JS-provided --
    # the main module must export it or the side module fails to link with
    # "tag import requires a WebAssembly.Tag".
    $jsProvided = '^(__indirect_function_table|__memory_base|__table_base|__stack_pointer|memory|' +
                  'invoke_.*|getTempRet0|setTempRet0|emscripten_asm_const.*|__cxa_find_matching_catch.*|' +
                  'llvm_eh_typeid_for|__resumeException|__THREW__|__asyncify_data|__asyncify_state|' +
                  '_ZN20__em_asm_sig_builder.*)$'

    $needed = $imports | Where-Object { $_ -notmatch $jsProvided } | ForEach-Object { "_$_" }
    Write-Host "auto-derived side-module imports: $($needed -join ',')" -ForegroundColor Yellow
    $exported += "," + ($needed -join ',')
}

$runtimeMethods = "ccall,cwrap,FS,UTF8ToString,stringToUTF8"
if ($MainModule -ne '0') { $runtimeMethods += ",loadDynamicLibrary" }

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
