import argparse
import os
import time

import requests


def response_to_json(response):
    response.raise_for_status()
    response_json = response.json()
    print(response_json)
    print("-" * 30)
    return response_json


def upload_apk_parse_json(post_url, apk_path, id, token):
    with open(apk_path, "rb") as apk_file:
        response = requests.post(post_url, files={"file": apk_file}, auth=(id, token))
    return response_to_json(response)


def browserstack_build_request(devices, app_url, test_suite_url, test_platform, id, token):
    headers = {}

    json_data = {
        "devices": devices,
        "app": app_url,
        "testSuite": test_suite_url,
    }

    build_response = requests.post(
        f"https://api-cloud.browserstack.com/app-automate/{test_platform}/v2/build",
        headers=headers,
        json=json_data,
        auth=(id, token),
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
        )

        test_response_json = response_to_json(test_response)
        tests_status = test_response_json["status"]

    return tests_status


if __name__ == "__main__":
    # handle cli args
    parser = argparse.ArgumentParser("Upload and run BrowserStack tests")

    parser.add_argument(
        "--test_platform", type=str, help="Testing platform", choices=["espresso", "xcuitest"], required=True
    )
    parser.add_argument("--app_apk_path", type=str, help="Path to the app APK", required=True)
    parser.add_argument("--test_apk_path", type=str, help="Path to the test suite APK", required=True)
    parser.add_argument(
        "--devices",
        type=str,
        nargs="+",
        help="List of devices to run the tests on. For more info, "
        "see https://www.browserstack.com/docs/app-automate/espresso/specify-devices",
        required=True,
    )

    args = parser.parse_args()

    browserstack_id = os.environ["BROWSERSTACK_ID"]
    browserstack_token = os.environ["BROWSERSTACK_TOKEN"]

    # Upload the app and test suites
    upload_app_json = upload_apk_parse_json(
        f"https://api-cloud.browserstack.com/app-automate/{args.test_platform}/v2/app",
        args.app_apk_path,
        browserstack_id,
        browserstack_token,
    )
    upload_test_json = upload_apk_parse_json(
        f"https://api-cloud.browserstack.com/app-automate/{args.test_platform}/v2/test-suite",
        args.test_apk_path,
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
