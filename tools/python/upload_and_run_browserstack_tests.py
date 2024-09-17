import argparse
import json
import time

import requests

parser = argparse.ArgumentParser("Upload and run BrowserStack tests")

parser.add_argument("--id", type=str, help="BrowserStack user ID")
parser.add_argument("--token", type=str, help="BrowserStack token")
parser.add_argument("--app_apk_path", type=str, help="Path to the app APK")
parser.add_argument("--test_apk_path", type=str, help="Path to the test suite APK")
# TODO: add link to browserstack documentation of available device strings you can pass in
parser.add_argument("--devices", type=str, nargs="+", help="List of devices to run the tests on")

args = parser.parse_args()


def post_response_to_json(response):
    if len(response) == 0:
        print("response received: ", response)
        raise Exception("No response from BrowserStack")
    try:
        json_response = json.loads(response)
        print("JSON response: ", json.dumps(json_response, indent=2))
        if "error" in json_response:
            raise Exception(json_response["error"])
        return json_response
    except Exception as e:
        print("response received: ", response)
        raise Exception("Invalid JSON response from BrowserStack") from e


def upload_apk_parse_json(post_url, apk_path):
    with open(apk_path, "rb") as apk_file:
        response = requests.post(post_url, files={"file": apk_file}, auth=(args.id, args.token))
    return post_response_to_json(response.text)


upload_app_json = upload_apk_parse_json(
    "https://api-cloud.browserstack.com/app-automate/espresso/v2/app", args.app_apk_path
)
upload_test_json = upload_apk_parse_json(
    "https://api-cloud.browserstack.com/app-automate/espresso/v2/test-suite", args.test_apk_path
)

headers = {}

json_data = {
    "devices": [
        "Samsung Galaxy S22-12.0",
    ],
    "app": upload_app_json["app_url"],
    "testSuite": upload_test_json["test_suite_url"],
}

build_response = requests.post(
    "https://api-cloud.browserstack.com/app-automate/espresso/v2/build",
    headers=headers,
    json=json_data,
    auth=(args.id, args.token),
)

build_response_json = post_response_to_json(build_response.text)

# GET TEST RESULTS

tests_status = "running"

while tests_status == "running":
    time.sleep(30)
    test_response = requests.get(
        "https://api-cloud.browserstack.com/app-automate/espresso/v2/builds/{build_id}".format(
            build_id=build_response_json["build_id"]
        ),
        auth=(args.id, args.token),
    )

    test_response_json = post_response_to_json(test_response.text)

    tests_status = test_response_json["status"]

print("=" * 30)
print(
    "Test suite details: ",
    "https://app-automate.browserstack.com/dashboard/v2/builds/" + build_response_json["build_id"],
)
print("=" * 30)
if tests_status == "failed":
    raise Exception("Tests failed. Go to the link above for more details.")
