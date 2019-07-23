# Gets the path to vcvarsall.bat for a particular Visual Studio version
# Run with -Verbose to see error output

param (
  [Parameter(Mandatory=$true)][string]$VsVersion
)

function ErrorExit([string]$message) {
  Write-Verbose $message
  Exit 1
}

$VS_VERSION_RANGE_MAP = @{
  "2017" = "[15.0,16.0)"
  "2019" = "[16.0,17.0)"
}
if (!$VS_VERSION_RANGE_MAP.Contains($VsVersion)) {
  $validVersions = $VS_VERSION_RANGE_MAP.Keys -join ", "
  ErrorExit("Invalid value for VsVersion. Allowed values: ${validVersions}")
}
$VsVersionRange = $VS_VERSION_RANGE_MAP[$VsVersion]

$VsWhereKnownPath = "${env:ProgramFiles(x86)}/Microsoft Visual Studio/Installer/vswhere.exe"
if (!(Test-Path -PathType Leaf -LiteralPath ${VsWhereKnownPath})) {
  ErrorExit("VsWhere.exe not found in expected location: ${VsWhereKnownPath}")
}

$VsInstallationPath = & ${VsWhereKnownPath} -latest -version ${VsVersionRange} -property installationPath -format value
if (!${?}) {
  ErrorExit("VsWhere.exe call failed.")
}
if ($VsInstallationPath.GetType() -ne [string]) {
  ErrorExit("Got unexpected output from VsWhere.exe.")
}

$VcVarsAllPath = "${VsInstallationPath}/VC/Auxiliary/Build/vcvarsall.bat"
if (!(Test-Path -PathType Leaf -LiteralPath ${VcVarsAllPath})) {
  ErrorExit("vcvarsall.bat not found in expected location: ${VcVarsAllPath}")
}

Write-Output $VcVarsAllPath
