# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

enum CMakeBuildType{
  Debug
  Release
  RelWithDebInfo
  MinSizeRel
}


# The DownloadAndExtract function was copied from: https://github.com/dotnet/arcade/blob/main/eng/common/native/CommonLibrary.psm1
<#
.SYNOPSIS
Get the name of a temporary folder under the native install directory
#>
function Get-TempDirectory {
  #TODO: what if the env does not exist?
  if (-not [string]::IsNullOrWhitespace($Env:AGENT_TEMPDIRECTORY)){
    return $Env:AGENT_TEMPDIRECTORY
  } else {
    return $Env:TEMP
  }
}

function Get-TempPathFilename {
  [CmdletBinding(PositionalBinding=$false)]
  Param (
    [Parameter(Mandatory=$True)]
    [string] $Path
  )
  $TempDir = Get-TempDirectory
  $TempFilename = Split-Path $Path -leaf
  $TempPath = Join-Path $TempDir $TempFilename
  return $TempPath
}

<#
.SYNOPSIS
Unzip an archive
.DESCRIPTION
Powershell module to unzip an archive to a specified directory
.PARAMETER ZipPath (Required)
Path to archive to unzip
.PARAMETER OutputDirectory (Required)
Output directory for archive contents
.PARAMETER Force
Overwrite output directory contents if they already exist
.NOTES
- Returns True and does not perform an extraction if output directory already exists but Overwrite is not True.
- Returns True if unzip operation is successful
- Returns False if Overwrite is True and it is unable to remove contents of OutputDirectory
- Returns False if unable to extract zip archive
#>
function Expand-Zip {
  [CmdletBinding(PositionalBinding=$false)]
  Param (
    [Parameter(Mandatory=$True)]
    [string] $ZipPath,
    [Parameter(Mandatory=$True)]
    [string] $OutputDirectory,
    [switch] $Force
  )
  if ([string]::IsNullOrWhitespace($OutputDirectory)){
     Write-Error "OutputDirectory cannot be empty"
  }
  Write-Host "Extracting '$ZipPath' to '$OutputDirectory'"
  try {
    if ((Test-Path $OutputDirectory) -And (-Not $Force)) {
      Write-Host "Directory '$OutputDirectory' already exists, skipping extract"
      return $True
    }
    if (Test-Path $OutputDirectory) {
      Write-Host "'Force' is 'True', but '$OutputDirectory' exists, removing directory"
      Remove-Item -Path $OutputDirectory -Force -Recurse
      if ($? -Eq $False) {
        Write-Error "Unable to remove '$OutputDirectory'"
        return $False
      }
    }

    $TempOutputDirectory = Join-Path "$(Split-Path -Parent $OutputDirectory)" "$(Split-Path -Leaf $OutputDirectory).tmp"

    if (Test-Path $TempOutputDirectory) {
      Remove-Item $TempOutputDirectory -Force -Recurse
    }
    New-Item -Path $TempOutputDirectory -Force -ItemType "Directory" | Out-Null

    Add-Type -assembly "system.io.compression.filesystem"
    [io.compression.zipfile]::ExtractToDirectory("$ZipPath", "$TempOutputDirectory")
    if ($? -Eq $False) {
      Write-Error "Unable to extract '$ZipPath'"
      return $False
    }

    Move-Item -Path $TempOutputDirectory -Destination $OutputDirectory
  }
  catch {
    Write-Host $_
    Write-Host $_.Exception

    return $False
  }
  return $True
}

<#
.SYNOPSIS
Helper module to install an archive to a directory
.DESCRIPTION
Helper module to download and extract an archive to a specified directory
.PARAMETER Uri
Uri of artifact to download
.PARAMETER InstallDirectory
Directory to extract artifact contents to.
.PARAMETER Force
Force download / extraction if file or contents already exist. Default = False
.PARAMETER DownloadRetries
Total number of retry attempts. Default = 5
.PARAMETER RetryWaitTimeInSeconds
Wait time between retry attempts in seconds. Default = 30
.NOTES
Returns False if download or extraction fail, True otherwise
#>
function DownloadAndExtract {
  [CmdletBinding(PositionalBinding=$false)]
  Param (
    [Parameter(Mandatory=$True)]
    [string] $Uri,
    [Parameter(Mandatory=$True)]
    [string] $InstallDirectory,
    [switch] $Force = $False,
    [int] $DownloadRetries = 5,
    [int] $RetryWaitTimeInSeconds = 30
  )
  # Define verbose switch if undefined
  $Verbose = $VerbosePreference -Eq "Continue"

  $TempToolPath = Get-TempPathFilename -Path $Uri
  Write-Host "TempToolPath=$TempToolPath"
  # Download native tool
  $DownloadStatus = Get-File -Uri $Uri `
                                           -Path $TempToolPath `
                                           -DownloadRetries $DownloadRetries `
                                           -RetryWaitTimeInSeconds $RetryWaitTimeInSeconds `
                                           -Force:$Force `
                                           -Verbose:$Verbose

  if ($DownloadStatus -Eq $False) {
    Write-Error "Download failed from $Uri"
    return $False
  }


  # Extract native tool
  $UnzipStatus = Expand-Zip -ZipPath $TempToolPath `
                                          -OutputDirectory $InstallDirectory `
                                          -Force:$Force `
                                          -Verbose:$Verbose
  if ($UnzipStatus -Eq $False) {
    # Retry Download one more time with Force=true
    $DownloadRetryStatus = Get-File -Uri $Uri `
                                             -Path $TempToolPath `
                                             -DownloadRetries 1 `
                                             -RetryWaitTimeInSeconds $RetryWaitTimeInSeconds `
                                             -Force:$True `
                                             -Verbose:$Verbose

    if ($DownloadRetryStatus -Eq $False) {
      Write-Error "Last attempt of download failed as well"
      return $False
    }

    # Retry unzip again one more time with Force=true
    $UnzipRetryStatus = Expand-Zip -ZipPath $TempToolPath `
                                            -OutputDirectory $InstallDirectory `
                                            -Force:$True `
                                            -Verbose:$Verbose
    if ($UnzipRetryStatus -Eq $False)
    {
      Write-Error "Last attempt of unzip failed as well"
      # Clean up partial zips and extracts
      if (Test-Path $TempToolPath) {
        Remove-Item $TempToolPath -Force
      }
      if (Test-Path $InstallDirectory) {
        Remove-Item $InstallDirectory -Force -Recurse
      }
      return $False
    }
  }

  return $True
}

<#
.SYNOPSIS
Download a file, retry on failure
.DESCRIPTION
Download specified file and retry if attempt fails
.PARAMETER Uri
Uri of file to download. If Uri is a local path, the file will be copied instead of downloaded
.PARAMETER Path
Path to download or copy uri file to
.PARAMETER Force
Overwrite existing file if present. Default = False
.PARAMETER DownloadRetries
Total number of retry attempts. Default = 5
.PARAMETER RetryWaitTimeInSeconds
Wait time between retry attempts in seconds Default = 30
#>
function Get-File {
  [CmdletBinding(PositionalBinding=$false)]
  Param (
    [Parameter(Mandatory=$True)]
    [string] $Uri,
    [Parameter(Mandatory=$True)]
    [string] $Path,
    [int] $DownloadRetries = 5,
    [int] $RetryWaitTimeInSeconds = 30,
    [switch] $Force = $False
  )
  $Attempt = 0

  if ($Force) {
    if (Test-Path $Path) {
      Remove-Item $Path -Force
    }
  }
  if (Test-Path $Path) {
    Write-Host "File '$Path' already exists, skipping download"
    return $True
  }

  $DownloadDirectory = Split-Path -ErrorAction Ignore -Path "$Path" -Parent
  if (-Not (Test-Path $DownloadDirectory)) {
    New-Item -path $DownloadDirectory -force -itemType "Directory" | Out-Null
  }

  $TempPath = "$Path.tmp"
  if (Test-Path -IsValid -Path $Uri) {
    Write-Verbose "'$Uri' is a file path, copying temporarily to '$TempPath'"
    Copy-Item -Path $Uri -Destination $TempPath
    Write-Verbose "Moving temporary file to '$Path'"
    Move-Item -Path $TempPath -Destination $Path
    return $?
  }
  else {
    # Don't display the console progress UI - it's a huge perf hit
    $ProgressPreference = 'SilentlyContinue'
    while($Attempt -Lt $DownloadRetries)
    {
      try {
        Invoke-WebRequest -UseBasicParsing -Uri $Uri -OutFile $TempPath
        Write-Verbose "Downloaded to temporary location '$TempPath'"
        Move-Item -Path $TempPath -Destination $Path
        Write-Verbose "Moved temporary file to '$Path'"
        return $True
      }
      catch {
        $Attempt++
        if ($Attempt -Lt $DownloadRetries) {
          $AttemptsLeft = $DownloadRetries - $Attempt
          Write-Warning "Download failed, $AttemptsLeft attempts remaining, will retry in $RetryWaitTimeInSeconds seconds"
          Start-Sleep -Seconds $RetryWaitTimeInSeconds
        }
        else {
          Write-Error $_
          Write-Error $_.Exception
        }
      }
    }
  }

  return $False
}

<#
    .Description
    The Get-DownloadURL function returns the download URL of a external dependency. The URL might be local file path or
     a remote HTTPS URL.

    .PARAMETER name
    The name of the dependency, should present in the first column of ONNX Runtime's deps.txt.

    .PARAMETER src_root
    The full path of ONNX Runtime's top level source diretory for locating deps.txt.
#>

function Get-DownloadURL {
    param (
        [Parameter(Mandatory)][string]$name,
        [Parameter(Mandatory)][string]$src_root
    )
    $entry = Import-Csv -Path "$src_root\cmake\deps.txt"  -Delimiter ';'  -Header name,url,hash -Encoding UTF8 | Where-Object name -eq $name
    return $entry.url
}

<#
    .Description
    The Install-Pybind function installs pybind11 headers, which are needed for building ONNX from source.

    .PARAMETER cmake_path
    The full path of cmake.exe

    .PARAMETER src_root
    The full path of ONNX Runtime's top level source diretory for locating deps.txt.

    .PARAMETER build_config
    The value of CMAKE_BUILD_TYPE, can be Debug, Release, RelWithDebInfo or MinSizeRel.
#>

function Install-Pybind {

    param (
        [Parameter(Mandatory)][string]$cmake_path,
        [Parameter(Mandatory)][string]$msbuild_path,
        [Parameter(Mandatory)][string]$src_root,
        [Parameter(Mandatory)][CMakeBuildType]$build_config,
        [Parameter(Mandatory)][string[]]$cmake_extra_args
    )

    pushd .

    $url=Get-DownloadURL -name pybind11 -src_root $src_root
    Write-Host "Downloading pybind11 from $url"
    $temp_dir = Get-TempDirectory
    $pybind_src_dir = Join-Path $temp_dir "pybind"
    $download_finished = DownloadAndExtract -Uri $url -InstallDirectory $pybind_src_dir -Force
    if(-Not $download_finished){
        Write-Host -Object "Download failed"
        exit 1
    }
    cd $pybind_src_dir
    cd *
    mkdir build
    cd build
    [string[]]$cmake_args = "..", "-DCMAKE_INSTALL_PREFIX=$install_prefix", "-DBUILD_TESTING=OFF"
    $cmake_args += $cmake_extra_args
    $p = Start-Process -FilePath $cmake_path -ArgumentList $cmake_args -NoNewWindow -Wait -PassThru
    $exitCode = $p.ExitCode
    if ($exitCode -ne 0) {
        Write-Host -Object "CMake command failed. Exitcode: $exitCode"
        exit $exitCode
    }

    $msbuild_args = "-nodeReuse:false", "-nologo", "-nr:false", "-maxcpucount", "-p:UseMultiToolTask=true", "-p:configuration=`"$build_config`""

    if ($use_cache) {
      $msbuild_args += "/p:CLToolExe=cl.exe /p:CLToolPath=C:\ProgramData\chocolatey\bin /p:TrackFileAccess=false /p:UseMultiToolTask=true"
    }

    $final_args = $msbuild_args + "pybind11.sln"
    Write-Host $final_args

    $p = Start-Process -FilePath $msbuild_path -ArgumentList $final_args -NoNewWindow -Wait -PassThru
    $exitCode = $p.ExitCode
    if ($exitCode -ne 0) {
        Write-Host -Object "Build pybind11.sln failed. Exitcode: $exitCode"
        exit $exitCode
    }

    $final_args = $msbuild_args + "INSTALL.vcxproj"
    $p = Start-Process -FilePath $msbuild_path -ArgumentList $final_args -NoNewWindow -Wait -PassThru
    $exitCode = $p.ExitCode
    if ($exitCode -ne 0) {
        Write-Host -Object "Install pybind failed. Exitcode: $exitCode"
        exit $exitCode
    }
    Write-Host "Installing pybind finished."

    popd
}

<#
    .Description
    The Install-ONNX function installs ONNX python package from source and also the python packages that it depends on.
    This script will build protobuf C/C++ lib/exe for the CPU arch of the current build machine, because we need to run
    protoc.exe on this machine.

    .PARAMETER cmake_path
    The full path of cmake.exe

    .PARAMETER src_root
    The full path of ONNX Runtime's top level source diretory

    .PARAMETER build_config
    The value of CMAKE_BUILD_TYPE, can be Debug, Release, RelWithDebInfo or MinSizeRel.
#>
function Install-Protobuf {

    param (
        [Parameter(Mandatory)][string]$cmake_path,
        [Parameter(Mandatory)][string]$msbuild_path,
        [Parameter(Mandatory)][string]$src_root,
        [Parameter(Mandatory)][CMakeBuildType]$build_config,
        [Parameter(Mandatory)][string[]]$cmake_extra_args
    )

    pushd .
    $url=Get-DownloadURL -name protobuf -src_root $src_root
    Write-Host "Downloading protobuf from $url"
    $temp_dir = Get-TempDirectory
    $protobuf_src_dir = Join-Path $temp_dir "protobuf"
    $download_finished = DownloadAndExtract -Uri $url -InstallDirectory $protobuf_src_dir -Force
    if(-Not $download_finished){
        exit 1
    }
    cd $protobuf_src_dir
    cd *
	# Search patch.exe
	$patch_path = 'C:\Program Files\Git\usr\bin\patch.exe'
	if(-not (Test-Path $patch_path -PathType Leaf)){
      $git_command_path = (Get-Command -CommandType Application git)[0].Path
      Write-Host "Git command path:$git_command_path"
      $git_installation_folder = Split-Path -Path (Split-Path -Path $git_command_path)
      $patch_path = Join-Path -Path $git_installation_folder "usr\bin\patch.exe"
	}
    if(Test-Path $patch_path -PathType Leaf){
      Write-Host "Patching protobuf ..."
      Get-Content $src_root\cmake\patches\protobuf\protobuf_cmake.patch | &$patch_path --ignore-whitespace -p1
    } else {
      Write-Host "Skip patching protobuf since we cannot find patch.exe at $patch_path"
    }

    # Run cmake to generate Visual Studio sln file
    [string[]]$cmake_args = ".", "-Dprotobuf_DISABLE_RTTI=ON", "-DCMAKE_BUILD_TYPE=$build_config", "-Dprotobuf_BUILD_TESTS=OFF", "-DBUILD_SHARED_LIBS=OFF", "-DCMAKE_PREFIX_PATH=$install_prefix",  "-DCMAKE_INSTALL_PREFIX=$install_prefix", "-Dprotobuf_MSVC_STATIC_RUNTIME=OFF"
    $cmake_args += $cmake_extra_args

    $p = Start-Process -FilePath $cmake_path -ArgumentList $cmake_args -NoNewWindow -Wait -PassThru
    $exitCode = $p.ExitCode
    if ($exitCode -ne 0) {
        Write-Host -Object "CMake command failed. Exitcode: $exitCode"
        exit $exitCode
    }

    $msbuild_args = "-nodeReuse:false", "-nologo", "-nr:false", "-maxcpucount", "-p:UseMultiToolTask=true", "-p:configuration=`"$build_config`""

    if ($use_cache) {
      $msbuild_args += "/p:CLToolExe=cl.exe /p:CLToolPath=C:\ProgramData\chocolatey\bin /p:TrackFileAccess=false /p:UseMultiToolTask=true"
    }

    $final_args = $msbuild_args + "protobuf.sln"
    Write-Host $final_args

    $p = Start-Process -FilePath $msbuild_path -ArgumentList $final_args -NoNewWindow -Wait -PassThru
    $exitCode = $p.ExitCode
    if ($exitCode -ne 0) {
        Write-Host -Object "Build protobuf.sln failed. Exitcode: $exitCode"
        exit $exitCode
    }

    $final_args = $msbuild_args + "INSTALL.vcxproj"
    $p = Start-Process -FilePath $msbuild_path -ArgumentList $final_args -NoNewWindow -Wait -PassThru
    $exitCode = $p.ExitCode
    if ($exitCode -ne 0) {
        Write-Host -Object "Install protobuf failed. Exitcode: $exitCode"
        exit $exitCode
    }
    Write-Host "Installing protobuf finished."
    popd
}

<#
    .Description
    The Install-ONNX function installs ONNX python package from source and also the python packages that it depends on.
    protoc.exe must exist in the PATH.
#>
function Install-ONNX {

    param (
        [Parameter(Mandatory)][CMakeBuildType]$build_config,
        [Parameter(Mandatory)][string]$src_root,
        [Parameter(Mandatory)][string]$protobuf_version
    )

    pushd .

    Write-Host "Installing python packages..."
    $p = Start-Process -NoNewWindow -Wait -PassThru -FilePath "python.exe" -ArgumentList "-m", "pip", "install", "--disable-pip-version-check", "setuptools", "wheel", "numpy", "protobuf==$protobuf_version"
    $exitCode = $p.ExitCode
    if ($exitCode -ne 0) {
        Write-Host -Object "Install dependent python wheels failed. Exitcode: $exitCode"
        exit $exitCode
    }

    $url=Get-DownloadURL -name onnx -src_root $src_root
    $temp_dir = Get-TempDirectory
    $onnx_src_dir = Join-Path $temp_dir "onnx"
    $download_finished = DownloadAndExtract -Uri $url -InstallDirectory $onnx_src_dir -Force
    if(-Not $download_finished){
        exit 1
    }
    cd $onnx_src_dir
    cd *
    [String]$requirements_txt_content = "protobuf==$protobuf_version`n"
    foreach($line in Get-Content '.\requirements.txt') {
      if($line -match "^protobuf"){
        Write-Host "Replacing protobuf version to $protobuf_version"
      } else{
        $requirements_txt_content += "$line`n"
      }
    }

    Set-Content -Path '.\requirements.txt' -Value $requirements_txt_content


    $Env:ONNX_ML=1
    if($build_config -eq 'Debug'){
       $Env:DEBUG='1'
    }
    $Env:CMAKE_ARGS="-DONNX_USE_PROTOBUF_SHARED_LIBS=OFF -DProtobuf_USE_STATIC_LIBS=ON -DONNX_USE_LITE_PROTO=OFF"

    $p = Start-Process -NoNewWindow -Wait -PassThru -FilePath "python.exe" -ArgumentList "setup.py", "bdist_wheel"
    $exitCode = $p.ExitCode
    if ($exitCode -ne 0) {
        Write-Host -Object "Generate wheel file failed. Exitcode: $exitCode"
        exit $exitCode
    }
    Write-Host "Uninstalling onnx and ignore errors if there is any..."
    python -m pip uninstall -y onnx -qq
    Write-Host "Installing the newly built ONNX python package"
    Get-ChildItem -Path dist/*.whl | foreach {
        $p = Start-Process -NoNewWindow -Wait -PassThru -FilePath "python.exe" -ArgumentList "-m", "pip", "--disable-pip-version-check", "install", "--upgrade", $_.fullname
        $exitCode = $p.ExitCode
        if ($exitCode -ne 0) {
          Write-Host -Object "Install wheel file failed. Exitcode: $exitCode"
          exit $exitCode
        }
    }
    Write-Host "Finished installing onnx"
    popd
}
