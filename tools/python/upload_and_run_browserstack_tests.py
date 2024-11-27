import argparse
import os
import sys
import time
from pathlib import Path

import requests

script_description = """
After building ONNXRuntime for Android or iOS, use this script to upload the app and test files to BrowserStack then
run the tests on the specified devices.

Find the Android test app in the repo here (as of rel-1.19.2):
java/src/test/android
"""


def response_to_json(response):
    response.raise_for_status()
    response_json = response.json()
    print(response_json)
    print("-" * 30)
    return response_json


def upload_apk_parse_json(post_url, apk_path, id, token):
    with open(apk_path, "rb") as apk_file:
        response = requests.post(post_url, files={"file": apk_file}, auth=(id, token), timeout=180)
    return response_to_json(response)


def browserstack_build_request(devices, app_url, test_suite_url, test_platform, id, token, project, build_tag):
    headers = {}

    json_data = {
        "devices": devices,
        "app": app_url,
        "testSuite": test_suite_url,
        "project": project,
        "buildTag": build_tag,
        "deviceLogs": True,
    }

    build_response = requests.post(
        f"https://api-cloud.browserstack.com/app-automate/{test_platform}/v2/build",
        headers=headers,
        json=json_data,
        auth=(id, token),
        timeout=180,
    )

    return response_to_json(build_response)


def build_query_loop(build_id, test_platform, id, token):
    """Called after a build is initiated on Browserstack. Repeatedly queries the REST endpoint to check for the
    status of the tests in 30 second intervals.
    """
    tests_status = "running"

    while tests_status == "running":
        time.sleep(30)

        test_response = requests.get(
            f"https://api-cloud.browserstack.com/app-automate/{test_platform}/v2/builds/{build_id}",
            auth=(id, token),
            timeout=30,
        )

        test_response_json = response_to_json(test_response)
        tests_status = test_response_json["status"]

    return tests_status


if __name__ == "__main__":
    # handle cli args
    parser = argparse.ArgumentParser(script_description)

    parser.add_argument(
        "--test_platform", type=str, help="Testing platform", choices=["espresso", "xcuitest"], required=True
    )
    parser.add_argument(
        "--app_path",
        type=Path,
        help=(
            "Path to the app file. "
            "For Android, typically, the app file (the APK) is in "
            "{build_output_dir}/android_test/android/app/build/outputs/apk/debug/app-debug.apk"
            ". For iOS, you will have to build an IPA file from the test app, which is built from the .xcarchive path"
        ),
        required=True,
    )
    parser.add_argument(
        "--test_path",
        type=Path,
        help=(
            "Path to the test suite file. "
            "Typically, the test APK is in "
            "{build_output_dir}/android_test/android/app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk"
            ". For iOS, you will have to create a .zip of the tests. After manually building the tests, the tests that you need to zip will be in {{Xcode DerivedData Folder Path}}/Build/Products"
        ),
        required=True,
    )
    parser.add_argument(
        "--devices",
        type=str,
        nargs="+",
        help="List of devices to run the tests on. For more info, "
        "see https://www.browserstack.com/docs/app-automate/espresso/specify-devices (Android) or https://www.browserstack.com/docs/app-automate/xcuitest/specify-devices (iOS)",
        required=True,
    )

    parser.add_argument(
        "--project",
        type=str,
        help="Identifier to logically group multiple builds together",
        default="ONNXRuntime tests",
    )
    parser.add_argument("--build_tag", type=str, help="Identifier to tag the build with a unique name", default="")
    args = parser.parse_args()

    try:
        browserstack_id = os.environ["BROWSERSTACK_ID"]
        browserstack_token = os.environ["BROWSERSTACK_TOKEN"]
    except KeyError:
        print("Please set the environment variables BROWSERSTACK_ID and BROWSERSTACK_TOKEN")
        print(
            "These values will be found at https://app-automate.browserstack.com/dashboard/v2 & clicking 'ACCESS KEY'"
        )
        sys.exit(1)

    # Upload the app and test suites
    upload_app_json = upload_apk_parse_json(
        f"https://api-cloud.browserstack.com/app-automate/{args.test_platform}/v2/app",
        args.app_path,
        browserstack_id,
        browserstack_token,
    )
    upload_test_json = upload_apk_parse_json(
        f"https://api-cloud.browserstack.com/app-automate/{args.test_platform}/v2/test-suite",
        args.test_path,
        browserstack_id,
        browserstack_token,
    )

    # Initiate build (send request to run the tests)
    build_response_json = browserstack_build_request(
        args.devices,
        upload_app_json["app_url"],
        upload_test_json["test_suite_url"],
        args.test_platform,
        browserstack_id,
        browserstack_token,
        args.project,
        args.build_tag,
    )

    # Get build status until the tests are no longer running
    tests_status = build_query_loop(
        build_response_json["build_id"], args.test_platform, browserstack_id, browserstack_token
    )

    test_suite_details_url = (
        f"https://app-automate.browserstack.com/dashboard/v2/builds/{build_response_json['build_id']}"
    )

    print("=" * 30)
    print("Test suite details: ", test_suite_details_url)
    print("=" * 30)
    if tests_status != "passed":
        raise Exception(f"Tests failed. Go to {test_suite_details_url} for more details.")
