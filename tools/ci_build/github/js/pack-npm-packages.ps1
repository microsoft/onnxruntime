# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# This script makes NPM packages for onnxruntime-common and onnxruntime-(node|web)
#
# Release Mode (release):
#   Do not update version number. Version number should be matching $(ORT_ROOT)/VERSION_NUMBER
#   Always generate packages for onnxruntime-common and onnxruntime-(node|web)
#
# Release Candidate Mode (rc):
#   Update version number to {VERSION_BASE}-rc.{YYYYMMDD}-{COMMIT}
#   Always generate packages for onnxruntime-common and onnxruntime-(node|web)
#
# Dev Mode (dev):
#   Compare current content with latest @dev package for onnxruntime-common. If no change, we
#   don't publish a new version of onnxruntime-common. Instead, we update the dependency version
#   to use the existing version.
#   Update version number to {VERSION_BASE}-dev.{YYYYMMDD}-{COMMIT}
#
# Custom Mode:
#   Use first commandline parameter as version number suffix.
#   Always generate packages for onnxruntime-common and onnxruntime-(node|web)
#

if ($Args.Count -ne 3) {
    throw "This script requires 3 arguments, but got $($Args.Count)"
}
$MODE=$Args[0] # "dev" or "release" or "rc"; otherwise it is considered as a version number
$ORT_ROOT=$Args[1] # eg. D:\source\onnxruntime
$TARGET=$Args[2] # "node" or "web"

Function Generate-Package-Version-Number {
    pushd $ORT_ROOT
    $version_base=Get-Content .\VERSION_NUMBER
    $version_timestamp=git show -s --format=%ct HEAD
    $version_commit=git rev-parse HEAD
    $version_commit_short=git rev-parse --short HEAD
    popd
    $version_date=[DateTimeOffset]::FromUnixTimeSeconds($version_timestamp).ToString("yyyyMMdd")
    if ($MODE -eq 'dev') {
        $version_number="$version_base-dev.$version_date-$version_commit_short"
    } elseif ($MODE -eq 'rc') {
        $version_number="$version_base-rc.$version_date-$version_commit_short"
    } elseif ($MODE -eq 'release')  {
        $version_number="$version_base"
    } else {
        $version_number="$version_base$MODE"
    }

    Write-Host "Generated version number for '$MODE': $version_number"
    return @{ version = $version_number; commit = $version_commit }
}

$JS_COMMON_DIR="$ORT_ROOT\js\common"
$JS_TARGET_DIR="$ORT_ROOT\js\$TARGET"

if ($MODE -eq "dev") {
    # For @dev builds, we compares the following 2 package versions for onnxruntime-common:
    # - 'current': the version we are currently building.
    # - 'latest': the latest @dev version from npm.js repository.
    #
    # If the contents of the 2 versions are identical, we don't publish a new version. Instead,
    # we only publish onnxruntime-node/onnxruntime-web and set its dependency's version to the
    # 'latest'.

    # check latest @dev version
    Write-Host "Start checking version for onnxruntime-common@dev"
    npm view onnxruntime-common@dev --json | Out-File -Encoding utf8 ./ort_common_latest_info.json
    $ort_common_latest_version=node -p "require('./ort_common_latest_info.json').version"
    $ort_common_latest_dist_tarball=node -p "require('./ort_common_latest_info.json').dist.tarball"
    $ort_common_latest_dist_shasum=node -p "require('./ort_common_latest_info.json').dist.shasum"
    Write-Host "latest version is: $ort_common_latest_version"

    # download package latest@dev
    Invoke-WebRequest $ort_common_latest_dist_tarball -OutFile ./latest.tgz
    if ($(Get-FileHash -Algorithm SHA1 .\latest.tgz).Hash -ne "$ort_common_latest_dist_shasum") {
        throw "SHASUM mismatch"
    }
    Write-Host "Tarball downloaded"

    # generate @dev version
    $version_number=Generate-Package-Version-Number

    # make package for latest
    pushd $JS_COMMON_DIR
    npm version $ort_common_latest_version
    echo $($version_number.commit) | Out-File -Encoding ascii -NoNewline -FilePath ./__commit.txt
    npm pack
    popd

    $current_tgz_compare_only="$JS_COMMON_DIR\onnxruntime-common-$ort_common_latest_version.tgz"
    if (!(Test-Path $current_tgz_compare_only)) {
        throw "File is not generated: $current_tgz_compare_only"
    }
    Write-Host "Package file ready for comparing: $current_tgz_compare_only"

    # extract packages
    mkdir ./latest
    pushd ./latest
    tar -xzf ../latest.tgz
    rm ../latest.tgz
    popd

    mkdir ./current
    pushd ./current
    tar -xzf $current_tgz_compare_only
    rm $current_tgz_compare_only
    popd

    # setup temp folder to compare
    mkdir ./temp
    pushd ./temp
    npm init -y
    npm install "dir-compare-cli@1.0.1" "json-diff@0.5.4"

    Write-Host "Compare package.json"
    npx json-diff ..\latest\package\package.json ..\current\package\package.json
    $use_latest=$?
    Write-Host "Result: $use_latest"
    if ($use_latest) {
        # package.json matches. now check package contents.

        # do not compare commit number
        if (test-path ../latest/package/__commit.txt) { rm ../latest/package/__commit.txt }
        if (test-path ../current/package/__commit.txt) { rm ../current/package/__commit.txt }
        # skip package.json, we already checked them
        rm ../latest/package/package.json
        rm ../current/package/package.json

        Write-Host "Compare package contents"
        npx dircompare -c ../latest/package/ ../current/package/
        $use_latest=$?
        Write-Host "Result: $use_latest"
    }

    popd

    if (!$use_latest) {
        Write-Host "Need update to onnxruntime-common@dev"
        # need to publish a new version for onnxruntime-common
        pushd $JS_COMMON_DIR
        npm version $($version_number.version)
        # file __commit.txt is already generated
        npm pack
        popd
    }

    # make package for target
    pushd $JS_TARGET_DIR
    npm version $($version_number.version)
    echo $($version_number.commit) | Out-File -Encoding ascii -NoNewline -FilePath ./__commit.txt
    npm pack
    popd
} elseif ($MODE -eq "release") {
    # release mode. always publish new package
    pushd $JS_COMMON_DIR
    npm pack
    popd
    pushd $JS_TARGET_DIR
    npm pack
    popd
} else {
    # release candidate mode or custom mode. always publish new package
    $version_number=Generate-Package-Version-Number

    pushd $JS_COMMON_DIR
    npm version $($version_number.version)
    npm pack
    popd

    pushd $JS_TARGET_DIR
    npm version $($version_number.version)
    npm pack
    popd
}
