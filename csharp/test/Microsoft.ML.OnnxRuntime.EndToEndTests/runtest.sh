#!/bin/bash
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

LocalNuGetRepo=$1
SourceRoot=$2
BuildDir=$3
TestDataUrl=$4
TestDataChecksum=$5

#TestDataUrl and TestDataChecksum comes from the build env variable

echo "Downloading test data"
python3 $SourceRoot/tools/ci_build/build.py --update --download_test_data --build_dir $BuildDir --test_data_url $TestDataUrl --test_data_checksum $TestDataChecksum
if [ $? -ne 0 ]; then
    echo "Failed to download test data"
    exit 1
fi

OldDir=`pwd`
cd $SourceRoot

MajorVersion=$(cat $SourceRoot/VERSION_NUMBER)
VersionSuffix=
if [ "$IsReleaseBuild" != "true" ]; then
    VersionSuffix=-dev-$(git rev-parse --short=8 HEAD)
fi
export CurrentOnnxRuntimeVersion=$MajorVersion$VersionSuffix
echo "Current NuGet package version is $CurrentOnnxRuntimeVersion"

dotnet restore $SourceRoot/csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests/Microsoft.ML.OnnxRuntime.EndToEndTests.csproj -s $LocalNuGetRepo -s https://api.nuget.org/v3/index.json
if [ $? -ne 0 ]; then
    echo "Failed to restore nuget packages for the test project"
    exit 1
fi

dotnet test $SourceRoot/csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests/Microsoft.ML.OnnxRuntime.EndToEndTests.csproj --no-restore --verbosity detailed
if [ $? -ne 0 ]; then
    echo "Failed to build or execute the end-to-end test"
    exit 1
fi

cd $OldDir