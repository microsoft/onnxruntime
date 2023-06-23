# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# This script makes NPM packages for onnxruntime-common and onnxruntime-(node|web|react-native)
#
# Release Mode (release):
#   Do not update version number. Version number should be matching $(ORT_ROOT)/VERSION_NUMBER
#   Always generate packages for onnxruntime-common and onnxruntime-(node|web|react-native)
#
# Release Candidate Mode (rc):
#   Update version number to {VERSION_BASE}-rc.{YYYYMMDD}-{COMMIT}
#   Always generate packages for onnxruntime-common and onnxruntime-(node|web|react-native)
#
# Dev Mode (dev):
#   Compare current content with latest @dev package for onnxruntime-common. If no change, we
#   don't publish a new version of onnxruntime-common. Instead, we update the dependency version
#   to use the existing version.
#   Update version number to {VERSION_BASE}-dev.{YYYYMMDD}-{COMMIT}
#
# Custom Mode:
#   Use first commandline parameter as version number suffix.
#   Always generate packages for onnxruntime-common and onnxruntime-(node|web|react-native)
#

if ($Args.Count -ne 3) {
    throw "This script requires 3 arguments, but got $($Args.Count)"
}
$MODE=$Args[0] # "dev" or "release" or "rc"; otherwise it is considered as a version number
$ORT_ROOT=$Args[1] # eg. D:\source\onnxruntime
$TARGET=$Args[2] # "node" or "web" or "react_native"

Function Generate-Package-Version-Number {
    pushd $ORT_ROOT
    $version_base=Get-Content ./VERSION_NUMBER
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

$JS_COMMON_DIR=Join-Path -Path "$ORT_ROOT" -ChildPath "js/common"
$JS_TARGET_DIR=Join-Path -Path "$ORT_ROOT" -ChildPath "js/$TARGET"

if ($MODE -eq "dev") {
    # For @dev builds, we compares the following 2 package versions for onnxruntime-common:
    # - 'current': the version we are currently building.
    # - 'latest': the latest @dev version from npm.js repository.
    #
    # If the contents of the 2 versions are identical, we don't publish a new version. Instead,
    # we only publish onnxruntime-node/onnxruntime-web/onnxruntime-react-native and
    # set its dependency's version to the 'latest'.

    # check latest @dev version
    Write-Host "Start checking version for onnxruntime-common@dev"
    npm view onnxruntime-common@dev --json | Out-File -Encoding utf8 ./ort_common_latest_info.json
    $ort_common_latest_version=node -p "require('./ort_common_latest_info.json').version"
    $ort_common_latest_dist_tarball=node -p "require('./ort_common_latest_info.json').dist.tarball"
    $ort_common_latest_dist_shasum=node -p "require('./ort_common_latest_info.json').dist.shasum"
    Write-Host "latest version is: $ort_common_latest_version"

    # download package latest@dev
    Invoke-WebRequest $ort_common_latest_dist_tarball -OutFile ./latest.tgz
    if ($(Get-FileHash -Algorithm SHA1 ./latest.tgz).Hash -ne "$ort_common_latest_dist_shasum") {
        throw "SHASUM mismatch"
    }
    Write-Host "Tarball downloaded"

    # generate @dev version
    $version_number=Generate-Package-Version-Number

    # make package for latest
    pushd $JS_COMMON_DIR
    npm version --allow-same-version $ort_common_latest_version
    echo $($version_number.commit) | Out-File -Encoding ascii -NoNewline -FilePath ./__commit.txt
    # update version.ts of common
    pushd ..
    npm run update-version common
    npm run format
    popd
    npm pack
    popd

    $current_tgz_compare_only=Join-Path -Path "$JS_COMMON_DIR" -ChildPath "onnxruntime-common-$ort_common_latest_version.tgz"
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
    $latest_package_json=Join-Path -Path ".." -ChildPath "latest/package/package.json"
    $current_package_json=Join-Path -Path ".." -ChildPath "current/package/package.json"
    npx json-diff $latest_package_json $current_package_json
    $use_latest=$?
    Write-Host "Result: $use_latest"
    if ($use_latest) {
        # package.json matches. now check package contents.

        # do not compare commit number
        $latest_package_commit=Join-Path -Path ".." -ChildPath "latest/package/__commit.txt"
        $current_package_commit=Join-Path -Path ".." -ChildPath "current/package/__commit.txt"
        if (test-path $latest_package_commit) { rm $latest_package_commit }
        if (test-path $current_package_commit) { rm $current_package_commit }
        # skip package.json, we already checked them
        rm $latest_package_json
        rm $current_package_json

        Write-Host "Compare package contents"
        $latest_package_dir=Join-Path -Path ".." -ChildPath "latest/package"
        $current_package_dir=Join-Path -Path ".." -ChildPath "current/package"
        npx dircompare -c $latest_package_dir $current_package_dir
        $use_latest=$?
        Write-Host "Result: $use_latest"
    }

    popd

    if (!$use_latest) {
        Write-Host "Need update to onnxruntime-common@dev"
        # need to publish a new version for onnxruntime-common
        pushd $JS_COMMON_DIR
        npm version --allow-same-version $($version_number.version)
        # file __commit.txt is already generated

        # update version.ts of common
        pushd ..
        npm run update-version common
        npm run format
        popd

        npm pack
        popd
    }

    # make package for target
    pushd $JS_TARGET_DIR
    npm version --allow-same-version $($version_number.version)
    echo $($version_number.commit) | Out-File -Encoding ascii -NoNewline -FilePath ./__commit.txt

    # update version.ts of TARGET
    pushd ..
    npm run update-version $TARGET
    npm run format
    popd

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
    npm version --allow-same-version $($version_number.version)

    # update version.ts of common
    pushd ..
    npm run update-version common
    npm run format
    popd

    npm pack
    popd

    pushd $JS_TARGET_DIR
    npm version --allow-same-version $($version_number.version)

    # update version.ts of TARGET
    pushd ..
    npm run update-version $TARGET
    npm run format
    popd

    npm pack
    popd
}
