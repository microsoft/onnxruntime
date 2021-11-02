# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# This script makes NPM packages for onnxruntime-common and onnxruntime-(node|web)
#
# Release Mode:
#   Do not update version number. Version number should be matching $(ORT_ROOT)/VERSION_NUMBER
#   Always generate packages for onnxruntime-common and onnxruntime-(node|web)
#
# Dev Mode:
#   Compare current content with latest @dev package for onnxruntime-common. If no change, we
#   don't publish a new version of onnxruntime-common. Instead, we update the dependency version
#   to use the existing version.
#

if ($Args.Count -ne 3) {
    Write-Error "This script requires 3 arguments, but got $($Args.Count)"
    exit 1
}
$MODE=$Args[0] # "dev" or "release"
$JS_ROOT=$Args[1] # eg. D:\source\onnxruntime\js
$TARGET=$Args[2] # "node" or "web"

$JS_COMMON_DIR="$JS_ROOT\common"
$JS_TARGET_DIR="$JS_ROOT\$TARGET"

if ($MODE -eq "dev") {
    # For @dev builds, we compares the following 2 package versions for onnxruntime-common:
    # - 'current': the version we are currently building.
    # - 'latest': the latest @dev version from npm.js repository.
    #
    # If the contents of the 2 versions are identical, we don't publish a new version. Instead,
    # we only publish onnxruntime-node/onnxruntime-web and set its dependency's version to the
    # 'latest'.

    # check latest @dev version
    npm view onnxruntime-common@dev --json | Out-File -Encoding utf8 ./ort_common_latest_info.json
    $ort_common_latest_version=node -p "require('./ort_common_latest_info.json').version"
    $ort_common_latest_dist_tarball=node -p "require('./ort_common_latest_info.json').dist.tarball"
    $ort_common_latest_dist_shasum=node -p "require('./ort_common_latest_info.json').dist.shasum"

    # download package latest@dev
    Invoke-WebRequest $ort_common_latest_dist_tarball -OutFile ./latest.tgz
    if ($(Get-FileHash -Algorithm SHA1 .\latest.tgz).Hash -ne "$ort_common_latest_dist_shasum") {
        Write-Error "SHASUM mismatch"
        exit 1
    }

    # make package for latest
    pushd $JS_COMMON_DIR
    npm version $ort_common_latest_version
    npm pack
    popd

    $current_tgz_compare_only="$JS_COMMON_DIR\onnxruntime-common-$ort_common_latest_version.tgz"
    if (!(Test-Path $current_tgz_compare_only)) {
        Write-Error "File is not generated: $current_tgz_compare_only"
        exit 1
    }

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

    # compare package.json
    npx json-diff ..\latest\package\package.json ..\current\package\package.json
    $use_latest=$?
    if ($use_latest) {
        # package.json matches. now check package contents.

        # do not compare commit number
        if (test-path ../latest/package/__commit.txt) { rm ../latest/package/__commit.txt }
        if (test-path ../current/package/__commit.txt) { rm ../current/package/__commit.txt }
        # skip package.json, we already checked them
        rm ../latest/package/package.json
        rm ../current/package/package.json

        # compare whole dictionary
        npx dircompare -c ../latest/package/ ../current/package/
        $use_latest=$?
    }

    popd

    # generate @dev version
    $dev_version_number=node $PSScriptRoot\generate-npm-package-dev-version.js

    if (!$use_latest) {
        # need to publish a new version for onnxruntime-common
        pushd $JS_COMMON_DIR
        npm version $dev_version_number
        npm pack
        popd
    }

    # make package for target
    pushd $JS_TARGET_DIR
    npm version $dev_version_number
    npm pack
    popd
} else {
    # release mode. always publish new package
    pushd $JS_COMMON_DIR
    npm pack
    popd
    pushd $JS_TARGET_DIR
    npm pack
    popd
}
