# See PythonTools.md in this folder

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Literal, TypedDict
from urllib.parse import urlparse

import requests
import yaml

# --- Data Structures for Type Hinting ---


class Variable(TypedDict, total=False):
    """Represents a pipeline variable. 'value' is optional for secrets."""

    value: str
    isSecret: bool


class ConfigurationRepository(TypedDict):
    """Represents the repository within the pipeline configuration."""

    fullName: str
    connection: dict[str, str]
    type: str


class PipelineConfiguration(TypedDict, total=False):
    """Represents the configuration part of a pipeline's details."""

    type: Literal["unknown", "yaml", "designerJson", "justInTime", "designerHyphenJson"]
    variables: dict[str, Variable]
    path: str
    repository: ConfigurationRepository


class PipelineDetails(TypedDict):
    """Represents the detailed data for a single pipeline."""

    id: int
    name: str
    url: str
    configuration: PipelineConfiguration


class EvaluationResult(TypedDict):
    """Represents the result of evaluating a single pipeline."""

    pipeline: PipelineDetails
    packaging_type: Literal["nightly", "release", "none"]
    has_pre_release_params: bool


# --- Configuration ---
ADO_ORGANIZATION = "aiinfra"
TARGET_REPO_URL = "https://github.com/microsoft/onnxruntime"
DAYS_SINCE_LAST_RUN = 30


def get_azure_cli_token() -> str:
    """Gets a token from Azure CLI for authentication."""
    ado_resource_id = "499b84ac-1321-427f-aa17-267ca6975798"
    command = [
        "az.cmd" if sys.platform == "win32" else "az",
        "account",
        "get-access-token",
        "--resource",
        ado_resource_id,
        "--query",
        "accessToken",
        "--output",
        "tsv",
    ]
    try:
        print("Attempting to get Azure DevOps token using the Azure CLI...")
        process = subprocess.run(command, capture_output=True, text=True, check=True, encoding="utf-8")
        token = process.stdout.strip()
        if not token:
            raise ValueError("Token from 'az' command is empty.")
        print("Successfully acquired Azure DevOps access token.")
        return token
    except FileNotFoundError:
        print("\nERROR: Azure CLI is not installed or not in your PATH.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"\nERROR during token acquisition: {e.stderr}")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR during token acquisition: {e}")
        sys.exit(1)


def get_pipelines(token: str, project: str) -> list[dict]:
    """Gets a summary list of all pipelines in the specified project."""
    print(f"\n--- Fetching Pipelines from project: {project} ---")
    headers = {"Authorization": f"Bearer {token}"}
    url = f"https://dev.azure.com/{ADO_ORGANIZATION}/{project}/_apis/pipelines?api-version=7.1-preview.1"
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        pipelines = response.json().get("value", [])
        print(f"Found {len(pipelines)} total pipelines.")
        return pipelines
    except requests.exceptions.RequestException as e:
        print(f"ERROR fetching pipelines: {e}")
        return []


def evaluate_single_pipeline(
    pipeline_summary: dict, token: str, branch: str, is_pr_build: bool
) -> EvaluationResult | None:
    """Fetches details for and evaluates a single pipeline against all criteria."""
    pipeline_name = pipeline_summary.get("name", "Unknown")
    pipeline_id = pipeline_summary.get("id")
    print(f"\nEvaluating pipeline: '{pipeline_name}' (ID: {pipeline_id})")

    headers = {"Authorization": f"Bearer {token}"}
    thirty_days_ago = datetime.utcnow() - timedelta(days=DAYS_SINCE_LAST_RUN)
    target_repo_path = urlparse(TARGET_REPO_URL).path.strip("/")
    has_pre_release_params = False

    try:
        detail_url = pipeline_summary.get("url")
        if not detail_url:
            print(f"  - SKIPPING: No detail URL found for '{pipeline_name}'.")
            return None

        detail_response = requests.get(detail_url, headers=headers)
        detail_response.raise_for_status()
        pipeline_details: PipelineDetails = detail_response.json()

        configuration = pipeline_details["configuration"]
        if configuration["type"] != "yaml":
            print(f"  - SKIPPING: Not a YAML-based pipeline (type is '{configuration['type']}').")
            return None

        repo_info = configuration["repository"]
        if repo_info["fullName"] != target_repo_path:
            repo_name = repo_info["fullName"]
            print(f"  - SKIPPING: Repository name '{repo_name}' does not match target '{target_repo_path}'.")
            return None

        base_detail_url = detail_url.split("?")[0]
        runs_url = f"{base_detail_url}/runs?$top=10&api-version=7.1-preview.1"
        runs_response = requests.get(runs_url, headers=headers)
        runs_response.raise_for_status()
        runs = runs_response.json().get("value", [])

        if not runs:
            print("  - SKIPPING: Pipeline has no previous runs.")
            return None

        last_run_time_str = next((run.get("finishedDate") for run in runs if run.get("finishedDate")), None)

        if not last_run_time_str:
            print("  - SKIPPING: No completed runs found in the last 10 attempts.")
            return None

        if "." in last_run_time_str:
            main_part, fractional_part = last_run_time_str.split(".", 1)
            if fractional_part.endswith("Z"):
                fractional_seconds = fractional_part[:-1]
                if len(fractional_seconds) > 6:
                    fractional_seconds = fractional_seconds[:6]
                last_run_time_str = f"{main_part}.{fractional_seconds}Z"

        last_run_time = datetime.strptime(last_run_time_str, "%Y-%m-%dT%H:%M:%S.%fZ")
        if last_run_time <= thirty_days_ago:
            print(
                f"  - SKIPPING: Last run was on {last_run_time.date()}, which is older than {DAYS_SINCE_LAST_RUN} days."
            )
            return None

        packaging_type: Literal["nightly", "release", "none"] = "none"
        if not is_pr_build:
            print(f"  - Checking YAML on branch '{branch}' for template and resource triggers...")
            yaml_path = configuration["path"]
            yaml_url = f"https://raw.githubusercontent.com/{repo_info['fullName']}/{branch}/{yaml_path}"

            yaml_response = requests.get(yaml_url)
            yaml_response.raise_for_status()
            yaml_content = yaml_response.text

            data = yaml.safe_load(yaml_content)
            if not isinstance(data, dict):
                print("  - WARNING: YAML content is not a dictionary. Skipping checks.")
                return None

            extends_block = data.get("extends", {})
            template_path = extends_block.get("template")
            expected_template = "v1/1ES.Official.PipelineTemplate.yml@1esPipelines"

            if template_path != expected_template:
                print(f"  - SKIPPING: YAML does not extend from the required template. Found: '{template_path}'.")
                return None
            print("  - OK: YAML extends from the correct template.")

            if (
                "resources" in data
                and isinstance(data.get("resources"), dict)
                and "pipelines" in data.get("resources", {})
            ):
                pipeline_list = data["resources"].get("pipelines", [])
                if isinstance(pipeline_list, list):
                    for p_resource in pipeline_list:
                        if isinstance(p_resource, dict) and "trigger" in p_resource:
                            print("  - SKIPPING: YAML contains a pipeline resource with a 'trigger' key.")
                            return None
            print("  - OK: No active pipeline resource trigger found in YAML.")

            # Check for packaging pipeline variables/parameters
            packaging_exceptions = ["onnxruntime-ios-packaging-pipeline"]
            if "packaging" in pipeline_name.lower() or "nuget" in pipeline_name.lower():
                print("  - Detected packaging pipeline. Checking for required variables/parameters...")
                if pipeline_name in packaging_exceptions:
                    print(f"  - OK: Allowing exception for '{pipeline_name}' which has no build mode flags.")
                    packaging_type = "none"
                elif "NIGHTLY_BUILD" in configuration.get("variables", {}):
                    packaging_type = "nightly"
                    print("  - OK: Found 'NIGHTLY_BUILD' pipeline variable.")
                elif any(p.get("name") == "IsReleaseBuild" for p in data.get("parameters", [])):
                    packaging_type = "release"
                    print("  - OK: Found 'IsReleaseBuild' YAML parameter.")
                else:
                    print(
                        f"  - SKIPPING: Packaging pipeline '{pipeline_name}' has neither a 'NIGHTLY_BUILD' variable nor an 'IsReleaseBuild' parameter."
                    )
                    return None

                # Check for pre-release parameters if it's a release packaging pipeline
                if packaging_type == "release":
                    yaml_params = data.get("parameters", [])
                    if isinstance(yaml_params, list):
                        param_names = {p.get("name") for p in yaml_params if isinstance(p, dict)}
                        if (
                            "PreReleaseVersionSuffixString" in param_names
                            and "PreReleaseVersionSuffixNumber" in param_names
                        ):
                            has_pre_release_params = True
                            print("  - OK: Found pre-release versioning parameters.")

        print(f"  - MATCH: '{pipeline_name}' matches all criteria.")
        return {
            "pipeline": pipeline_details,
            "packaging_type": packaging_type,
            "has_pre_release_params": has_pre_release_params,
        }

    except KeyError as e:
        print(f"  - SKIPPING '{pipeline_name}': Missing expected key {e} in pipeline details.")
    except requests.exceptions.RequestException as e:
        print(f"  - ERROR processing '{pipeline_name}': {e}")
    except yaml.YAMLError as e:
        print(f"  - WARNING for '{pipeline_name}': Could not parse YAML file: {e}. Skipping.")
    except Exception as e:
        print(f"  - ERROR processing '{pipeline_name}': An unexpected error occurred: {e}")

    return None


def filter_pipelines(pipelines: list[dict], token: str, branch: str, is_pr_build: bool) -> list[EvaluationResult]:
    """Filters pipelines based on specified criteria by processing them in parallel."""
    print("\n--- Filtering Pipelines in Parallel ---")
    filtered_results: list[EvaluationResult] = []

    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_pipeline = {
            executor.submit(evaluate_single_pipeline, p, token, branch, is_pr_build): p for p in pipelines
        }
        for future in as_completed(future_to_pipeline):
            result = future.result()
            if result:
                filtered_results.append(result)

    print(f"\nFound {len(filtered_results)} pipelines to trigger:")
    for result in filtered_results:
        print(f"  - {result['pipeline']['name']}")
    return filtered_results


def cancel_running_builds(pipeline_id: int, branch: str, token: str, project: str):
    """Finds and cancels any running builds for a given pipeline and branch."""
    print(
        f"\n--- Checking for running builds for Pipeline ID: {pipeline_id} on branch '{branch}' in project '{project}' ---"
    )
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    builds_url = f"https://dev.azure.com/{ADO_ORGANIZATION}/{project}/_apis/build/builds?definitions={pipeline_id}&branchName={branch}&statusFilter=inProgress,notStarted&api-version=7.1"

    try:
        response = requests.get(builds_url, headers=headers)
        response.raise_for_status()
        running_builds = response.json().get("value", [])

        if not running_builds:
            print("No running builds found to cancel.")
            return

        for build in running_builds:
            build_id = build["id"]
            print(f"Cancelling running build ID: {build_id}...")
            cancel_url = (
                f"https://dev.azure.com/{ADO_ORGANIZATION}/{project}/_apis/build/builds/{build_id}?api-version=7.1"
            )
            payload = {"status": "cancelling"}

            patch_response = requests.patch(cancel_url, headers=headers, data=json.dumps(payload))
            patch_response.raise_for_status()
            print(f"Successfully requested cancellation for build ID: {build_id}.")

    except requests.exceptions.RequestException as e:
        print(f"ERROR checking or cancelling running builds: {e}")


def trigger_pipeline(
    pipeline_id: int,
    token: str,
    branch: str,
    project: str,
    nightly_override: str | None,
    release_override: str | None,
    packaging_type: str,
    has_pre_release_params: bool,
    pre_release_suffix_string: str | None,
    pre_release_suffix_number: int | None,
) -> int | None:
    """Triggers a pipeline and returns the new build ID."""
    print(f"\n--- Triggering Pipeline ID: {pipeline_id} on branch '{branch}' in project '{project}' ---")
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    run_url = f"https://dev.azure.com/{ADO_ORGANIZATION}/{project}/_apis/pipelines/{pipeline_id}/runs?api-version=7.1-preview.1"

    payload: dict[str, any] = {"resources": {"repositories": {"self": {"refName": branch}}}}
    template_params: dict[str, any] = {}

    if nightly_override is not None and packaging_type == "nightly":
        print(f"Overriding NIGHTLY_BUILD variable to '{nightly_override}'.")
        payload["variables"] = {"NIGHTLY_BUILD": {"value": nightly_override}}
    elif release_override is not None and packaging_type == "release":
        print(f"Overriding IsReleaseBuild parameter to '{release_override}'.")
        template_params["IsReleaseBuild"] = release_override

    # Add pre-release parameters if the pipeline supports them and user provided them
    if has_pre_release_params:
        if pre_release_suffix_string is not None:
            print(f"Setting PreReleaseVersionSuffixString parameter to '{pre_release_suffix_string}'.")
            template_params["PreReleaseVersionSuffixString"] = pre_release_suffix_string
        if pre_release_suffix_number is not None:
            print(f"Setting PreReleaseVersionSuffixNumber parameter to {pre_release_suffix_number}.")
            template_params["PreReleaseVersionSuffixNumber"] = pre_release_suffix_number

    if template_params:
        payload["templateParameters"] = template_params

    try:
        response = requests.post(run_url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        build_info = response.json()
        build_id = build_info.get("id")
        print(f"Successfully triggered build. Build ID: {build_id}")
        print(f"    Web URL: {build_info.get('_links', {}).get('web', {}).get('href')}")
        return build_id
    except requests.exceptions.RequestException as e:
        print(f"ERROR triggering pipeline: {e}")
        if e.response is not None:
            print(f"    Response: {e.response.text}")
        return None


def main():
    """Main function to orchestrate the pipeline triggering process."""
    parser = argparse.ArgumentParser(description="Trigger Azure DevOps pipelines based on specific criteria.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print pipelines that would be triggered, but don't actually trigger them.",
    )
    parser.add_argument("--branch", default="main", help="The target branch for CI builds. Defaults to 'main'.")
    parser.add_argument("--pr", type=int, help="The pull request ID to trigger PR builds for. This overrides --branch.")
    parser.add_argument(
        "--build-mode",
        choices=["nightly", "release"],
        help="Specify the build mode for packaging pipelines (nightly or release). This sets NIGHTLY_BUILD and IsReleaseBuild accordingly.",
    )
    parser.add_argument(
        "--no-cancel",
        action="store_true",
        dest="no_cancel_builds",
        help="Do not cancel existing running builds for the pipeline before triggering a new one.",
    )
    # New arguments for pre-release versioning
    parser.add_argument(
        "--pre-release-suffix-string",
        choices=["alpha", "beta", "rc", "none"],
        help="Suffix for pre-release versions (e.g., 'rc'). Requires the pipeline to have the parameter.",
    )
    parser.add_argument(
        "--pre-release-suffix-number",
        type=int,
        help="Number for pre-release versions (e.g., '1'). Requires the pipeline to have the parameter.",
    )

    args = parser.parse_args()

    project = "Lotus"
    branch_for_trigger = f"refs/heads/{args.branch}"
    branch_for_yaml_fetch = args.branch
    is_pr_build = False

    if (args.pre_release_suffix_string and not args.pre_release_suffix_number) or (
        not args.pre_release_suffix_string and args.pre_release_suffix_number
    ):
        parser.error("--pre-release-suffix-string and --pre-release-suffix-number must be used together.")

    if args.pr:
        print(f"--- Pull Request Mode Activated for PR #{args.pr} ---")
        project = "PublicPackages"
        branch_for_trigger = f"refs/pull/{args.pr}/head"
        branch_for_yaml_fetch = args.branch
        is_pr_build = True
    else:
        print(f"--- CI Build Mode Activated for branch '{args.branch}' ---")

    if args.dry_run:
        print("DRY RUN MODE ENABLED: No pipelines will be triggered.")

    token = get_azure_cli_token()
    all_pipelines = get_pipelines(token, project)
    if not all_pipelines:
        print("\nERROR: Could not retrieve any pipelines. Exiting.")
        return

    pipelines_to_trigger = filter_pipelines(all_pipelines, token, branch_for_yaml_fetch, is_pr_build)

    if not pipelines_to_trigger:
        print("\nNo pipelines matched the specified criteria. No builds will be triggered.")
        return

    if args.dry_run:
        print(f"\n--- The following pipelines would be triggered on branch '{branch_for_trigger}' ---")
        for result in pipelines_to_trigger:
            print(f"  - {result['pipeline']['name']} (ID: {result['pipeline']['id']})")
    else:
        print(f"\n--- Triggering {len(pipelines_to_trigger)} Pipelines on branch '{branch_for_trigger}' ---")
        nightly_override = None
        release_override = None
        if args.build_mode == "nightly":
            nightly_override = "1"
            release_override = "false"
        elif args.build_mode == "release":
            nightly_override = "0"
            release_override = "true"

        # If pre-release flags are used, it implies a release build.
        if args.pre_release_suffix_string:
            print("Pre-release suffix provided. Forcing 'release' build mode.")
            if args.build_mode and args.build_mode != "release":
                print(f"Warning: --build-mode={args.build_mode} is overridden by pre-release flags.")
            nightly_override = "0"
            release_override = "true"

        for result in pipelines_to_trigger:
            pipeline = result["pipeline"]
            packaging_type = result["packaging_type"]
            has_pre_release_params = result["has_pre_release_params"]

            if not args.no_cancel_builds:
                cancel_running_builds(pipeline["id"], branch_for_trigger, token, project)
            else:
                print(f"\nSkipping cancellation for Pipeline ID: {pipeline['id']} as per --no-cancel flag.")

            trigger_pipeline(
                pipeline_id=pipeline["id"],
                token=token,
                branch=branch_for_trigger,
                project=project,
                nightly_override=nightly_override,
                release_override=release_override,
                packaging_type=packaging_type,
                has_pre_release_params=has_pre_release_params,
                pre_release_suffix_string=args.pre_release_suffix_string,
                pre_release_suffix_number=args.pre_release_suffix_number,
            )


if __name__ == "__main__":
    main()

