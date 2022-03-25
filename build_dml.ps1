$buildArgs = @()
$buildArgs += "--build_dir $PSScriptRoot\build\Windows"
$buildArgs += '--config Debug'
$buildArgs += '--build_shared_lib'
$buildArgs += '--parallel'
$buildArgs += '--use_dml'
$buildArgs += '--skip_tests'
$buildArgs += '--skip_submodule_sync'
$buildArgs += '--target install'
$buildArgs += "--cmake_extra_defines CMAKE_INSTALL_PREFIX=$PSScriptRoot/build/install --target install"

python $PSScriptRoot\tools\ci_build\build.py "$buildArgs"