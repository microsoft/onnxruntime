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
    "--gdk",
    "--gdk_edition", "210602",
    "--cmake_extra_defines", 
        "CMAKE_INSTALL_PREFIX=$PSScriptRoot\build\install", 
        "onnxruntime_USE_CUSTOM_DIRECTML=ON", 
        "dml_INCLUDE_DIR=`"$DmlInstallPath/include`""
)

python $PSScriptRoot\tools\ci_build\build.py $pythonArgs