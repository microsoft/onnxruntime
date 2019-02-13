pushd client

msbuild client.vcxproj /p:Platform=x64 /p:Configuration=Debug
msbuild client.vcxproj /p:Platform=x64 /p:Configuration=Release

docker build --rm -t brainslice-client-build:latest .
docker run --rm -v %~dp0/client:/client -v %DevKit%:/devkit -e DevKit=/devkit -v %PkgBond_Cpp%:/bond -e PkgBond_Cpp=/bond brainslice-client-build:latest make Configuration=Debug
docker run --rm -v %~dp0/client:/client -v %DevKit%:/devkit -e DevKit=/devkit -v %PkgBond_Cpp%:/bond -e PkgBond_Cpp=/bond brainslice-client-build:latest make Configuration=Release

nuget pack BrainSlice.v3.Client.nuspec -Properties DevKit=%DevKit%
popd

pushd firmware
msbuild firmware.proj /p:Platform=x64 /p:Configuration=Release
popd
