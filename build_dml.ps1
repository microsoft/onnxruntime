# This file won't get merged upstream. It's for convenience only.

param
(
    [string]$DmlInstallPath = 'S:/dml_gdk210602'
)

$pythonArgs = @(
    "--build_dir", "$PSScriptRoot\build\Windows",
    "--config", "Debug",
    "--build_shared_lib",
    "--parallel",
    "--skip_tests",
    "--skip_submodule_sync",
    "--target", "install",
    "--use_dml",
    "--use_gdk",
    "--gdk_edition", "210602",
    "--dml_path", "s:/dml_gdk210602"
    "--cmake_extra_defines", "CMAKE_INSTALL_PREFIX=$PSScriptRoot\build\install"
)

python $PSScriptRoot\tools\ci_build\build.py $pythonArgs