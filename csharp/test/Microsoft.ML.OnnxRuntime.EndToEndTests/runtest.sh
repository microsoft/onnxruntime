#!/bin/bash
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

LocalNuGetRepo=$1
# Assumes: working dir = $(Build.SourcesDirectory/csharp)

MajorVersion=$(cat ../../../VERSION_NUMBER)
VersionSuffix=
if [ "$IsReleaseBuild" != "true" ]; then
    VersionSuffix = -dev-$(git rev-parse --short HEAD)
fi
export CurrentOnnxRuntimeVersion=$MajorVersion$VersionSuffix
echo "Current NuGet package version is $CurrentOnnxRuntimeVersion"

dotnet restore ./test/Microsoft.ML.OnnxRuntime.EndToEndTests/Microsoft.ML.OnnxRuntime.EndToEndTests.csproj -s $LocalNuGetRepo --configfile ./Nuget.CSharp.config
if [ $? -ne 0 ]
    echo "Failed to restore nuget packages for the test project"
    exit 1
)

dotnet test ./test/Microsoft.ML.OnnxRuntime.EndToEndTests/Microsoft.ML.OnnxRuntime.EndToEndTests.csproj --no-restore
if [ $? -ne 0 ]
    echo "Failed to build or execute the end-to-end test"
    exit 1
)