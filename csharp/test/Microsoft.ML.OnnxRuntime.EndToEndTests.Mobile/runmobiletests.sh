#!/bin/bash

parametersResolved=true
parametersValid=true
locallatestAndroidTestRunId=null
latestiOSTestRunId=null
latestAndroidTestRunState=null
latestiOSTestRunState=null
androidTestResultStatus=null
iosTestResultStatus=null
finishedTests=false
errors=0
maxErrors=10
timeoutInSeconds=60
failed=false

# Determine latest test runs (first item is latest run)
function getLatestTestRunIds() {
    idRegex='"id":"([0-9a-z-]*)"'

    androidTestRunsJson=$(curl -s -b -v -w "%{http_code}" -H "X-API-Token:$tokenAndroid" "https://api.appcenter.ms/v0.1/apps/$org/$appNameAndroid/test_runs")
    [[ $androidTestRunsJson =~ $idRegex ]]   
    latestAndroidTestRunId="${BASH_REMATCH[1]}"

    iosTestRunsJson=$(curl -s -b -v -w "%{http_code}" -H "X-API-Token:$tokeniOS" "https://api.appcenter.ms/v0.1/apps/$org/$appNameiOS/test_runs")
    [[ $iosTestRunsJson =~ $idRegex ]]   
    latestiOSTestRunId="${BASH_REMATCH[1]}"
}

# Check status of latest test runs
function checkTestRunStatus() {
    runStatusRegex='"runStatus":"([a-z]*)"'
    testResultRegex='"resultStatus":"([a-z]*)"'

    # Get Android and iOS test run id's if they have not already been resolved
    [ $latestiOSTestRunId = null ] || [ $latestAndroidTestRunId = null ] && { getLatestTestRunIds; }

    # Error if it's not possible to resolve both test run ID values
    [ $latestiOSTestRunId = null ] || [ $latestAndroidTestRunId = null ] && { ((errors++)); return; }
    
    androidLatestTestRunJson=$(curl -s -b -v -w "%{http_code}" -H "X-API-Token:$tokenAndroid" "https://api.appcenter.ms/v0.1/apps/$org/$appNameAndroid/test_runs/$latestAndroidTestRunId")   
    [[ $androidLatestTestRunJson =~ $runStatusRegex ]]   
    latestAndroidTestRunState="${BASH_REMATCH[1]}"

    iosLatestTestRunJson=$(curl -s -b -v -w "%{http_code}" -H "X-API-Token:$tokeniOS" "https://api.appcenter.ms/v0.1/apps/$org/$appNameiOS/test_runs/$latestiOSTestRunId")
    [[ $iosLatestTestRunJson =~ $runStatusRegex ]]   
    latestiOSTestRunState="${BASH_REMATCH[1]}"

    if [ "$latestAndroidTestRunState" = null ] || [ "$latestiOSTestRunState" = null ]; then
        ((errors++))
        return
    elif [ "$latestAndroidTestRunState" = "finished" ] && [ "$latestiOSTestRunState" = "finished" ]; then
        finishedTests=true
        [[ $androidLatestTestRunJson =~ $testResultRegex ]]   
        androidTestResultStatus="${BASH_REMATCH[1]}"
        [[ $iosLatestTestRunJson =~ $testResultRegex ]]   
        iosTestResultStatus="${BASH_REMATCH[1]}"
    else
        finishedTests=false
    fi
}

echo ""
echo "========== Platform-Specific Testing Started =========="
echo ""

# Resolve parameters
for i in "$@"; do
    case $1 in
        "" ) break ;;
        -b | --test-build-dir ) sourceDirectory="$2"; shift ;;
        -p | --packages-dir ) packagesDirectory="$2"; shift ;;
        -a | --apk ) androidApp="$2"; shift ;;
        -i | --ipa ) iosApp="$2"; shift ;;
        -o | --org ) org="$2"; shift ;;
        -na | --app-name-android ) appNameAndroid="$2"; shift ;;
        -ni | --app-name-ios ) appNameiOS="$2"; shift ;;
        -ta | --token-android ) tokenAndroid="$2"; shift ;;
        -ti | --token-ios ) tokeniOS="$2"; shift ;;
        -* | --*) echo "Unknown option: '$1'"; exit 1 ;;
        * ) echo "Unknown argument: '$1'"; exit 1 ;;
    esac
    shift
done

# Validate parameters have been resolved
[ -z "$sourceDirectory" ] && { echo "Missing --source-directory parameter"; parametersResolved=false; }
[ -z "$packagesDirectory" ] && { echo "Missing --packages-directory parameter"; parametersResolved=false; }
[ -z "$androidApp" ] && { echo "Missing --android-app parameter"; parametersResolved=false; }
[ -z "$iosApp" ] && { echo "Missing --ios-app parameter"; parametersResolved=false; }
[ -z "$org" ] && { echo "Missing --org parameter"; parametersResolved=false; }
[ -z "$appNameAndroid" ] && { echo "Missing --app-name-android parameter"; parametersResolved=false; }
[ -z "$appNameiOS" ] && { echo "Missing --app-name-ios parameter"; parametersResolved=false; }
[ -z "$tokenAndroid" ] && { echo "Missing --token-android parameter"; parametersResolved=false; }
[ -z "$tokeniOS" ] && { echo "Missing --token-ios parameter"; parametersResolved=false; }

[ $parametersResolved = false ] && {
    echo ""
    echo "========== Platform-Specific Testing Completed =========="
    echo ""
    exit 1
}

# Validate parameter values are valid
[ ! -d "$sourceDirectory" ] && { echo "No directory exists at path specified for --test-build-dir"; parametersValid=false; }
[ ! -d "$packagesDirectory" ] && { echo "No directory exists at path specified for --packages-dir"; parametersValid=false; }
[ ! -f "$androidApp" ] && { echo "No apk file found using filepath specified for --apk"; parametersValid=false; }
[ ! -f "$iosApp" ] && { echo "No ipa file found using filepath specified for --ipa"; parametersValid=false; }

toolsDir="$(find "$packagesDirectory" -name 'tools' ! -name 'test-cloud.exe' | head -1)"
[ ! -d "$toolsDir" ] && { echo "Unable to locate the requisite tools directory within the directory specified for --packages-dir"; parametersValid=false; }

[ $parametersValid = false ] && {
    echo ""
    echo "========== Platform-Specific Testing Completed =========="
    echo ""
    exit 1
}

# Start the tests
echo "Starting Tests"

# Start Android Tests
startAndroidTests=$(appcenter test run uitest \
    --app "$org/$appNameAndroid" \
    --devices 3aaf6e5b \
    --app-path $androidApp \
    --test-series "platformunittests" \
    --locale "en_US" \
    --build-dir $sourceDirectory \
    --uitest-tools-dir $toolsDir \
    --async)

echo ""
echo "  Android: STARTED"

# Start iOS Tests
startiOSTests=$(appcenter test run uitest \
    --app "$org/$appNameiOS" \
    --devices f236dfc0 \
    --app-path $iosApp \
    --test-series "platformunittests" \
    --locale "en_US" \
    --build-dir $sourceDirectory \
    --uitest-tools-dir $toolsDir \
    --async)

echo "      iOS: STARTED"

# Monitor state of latest test runs (todo: should cancel any test runs successfully started if one or more failed to start )
checkTestRunStatus

[ "$latestAndroidTestRunState" != "running" ] && [ "$latestiOSTestRunState" != "running" ] && { 
    echo ""
    echo "There was an issue starting the tests"
    echo ""
    echo "========== Platform-Specific Testing Completed =========="
    echo ""
    exit 1
}

echo ""
echo "Monitoring Test Runs" 
echo ""
echo "  Android: $latestAndroidTestRunId"
echo "      iOS: $latestiOSTestRunId"

while [ $finishedTests = false ] && [ $failed = false ]; do
    sleep $timeoutInSeconds
    checkTestRunStatus

	if [ "$errors" -ge "$maxErrors" ]; then
        failed=true
	fi
done

# Determine test outcome
[ $failed = true ] && { 
    echo ""
    echo "There was an issue monitoring the tests"
    echo ""
    echo "========== Platform-Specific Testing Completed =========="
    echo ""
    exit 1
}

echo ""
echo "Tests Finished"
echo ""

[ "$androidTestResultStatus" = "passed" ] && { echo "  Android: PASS"; }
[ "$androidTestResultStatus" != "passed" ] && { echo "  Android: FAIL"; }
[ "$iosTestResultStatus" = "passed" ] && { echo "      iOS: PASS"; }
[ "$iosTestResultStatus" != "passed" ] && { echo "      iOS: FAIL"; }

echo ""
echo "========== Platform-Specific Testing Completed =========="
echo ""

[ "$androidTestResultStatus" != "passed" ] && [ "$iosTestResultStatus" != "passed" ] && { exit 1; }