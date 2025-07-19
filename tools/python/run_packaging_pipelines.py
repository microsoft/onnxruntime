# This script is designed to trigger specific Azure DevOps pipelines based on a set of criteria.
#
# It supports two modes:
#
# 1. CI Build Mode (Default):
#    - Triggers pipelines in the 'Lotus' project.
#    - Filters pipelines based on the following criteria:
#      - Repository Association: Must be 'https://github.com/microsoft/onnxruntime'.
#      - Recent Activity: Must have run in the last 30 days.
#      - Pipeline Type: Must be YAML-based.
#      - Trigger Type: Must NOT be triggered by another pipeline resource.
#      - Template Requirement: Must extend from 'v1/1ES.Official.PipelineTemplate.yml@1esPipelines'.
#
# 2. Pull Request (PR) Mode:
#    - Activated by using the '--pr <ID>' argument.
#    - Triggers pipelines in the 'PublicPackages' project.
#    - Filters pipelines based on a simplified criteria:
#      - Repository Association: Must be 'https://github.com/microsoft/onnxruntime'.
#      - Recent Activity: Must have run in the last 30 days.
#      - Pipeline Type: Must be YAML-based.
#
# The script also includes a feature to cancel any currently running builds for a matching
# pipeline on the target branch/PR before queuing a new one.

import subprocess
import sys
import requests
import json
import argparse
import yaml
from datetime import datetime, timedelta
from typing import TypedDict, List, Dict, Optional, Literal
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Data Structures for Type Hinting ---

class Variable(TypedDict, total=False):
    """Represents a pipeline variable. 'value' is optional for secrets."""
    value: str
    isSecret: bool

class ConfigurationRepository(TypedDict):
    """Represents the repository within the pipeline configuration."""
    fullName: str
    connection: Dict[str, str]
    type: str

class PipelineConfiguration(TypedDict, total=False):
    """Represents the configuration part of a pipeline's details."""
    type: Literal["unknown", "yaml", "designerJson", "justInTime", "designerHyphenJson"]
    variables: Dict[str, Variable]
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


# --- Configuration ---
ADO_ORGANIZATION = "aiinfra"
TARGET_REPO_URL = "https://github.com/microsoft/onnxruntime"
DAYS_SINCE_LAST_RUN = 30

def get_azure_cli_token() -> str:
    """Gets a token from Azure CLI for authentication."""
    ado_resource_id = "499b84ac-1321-427f-aa17-267ca6975798"
    command = ["az.cmd" if sys.platform == "win32" else "az", "account", "get-access-token", "--resource", ado_resource_id, "--query", "accessToken", "--output", "tsv"]
    try:
        print("Attempting to get Azure DevOps token using the Azure CLI...")
        process = subprocess.run(command, capture_output=True, text=True, check=True, encoding='utf-8')
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

def get_pipelines(token: str, project: str) -> List[Dict]:
    """Gets a summary list of all pipelines in the specified project."""
    print(f"\n--- Fetching Pipelines from project: {project} ---")
    headers = {"Authorization": f"Bearer {token}"}
    url = f"https://dev.azure.com/{ADO_ORGANIZATION}/{project}/_apis/pipelines?api-version=7.1-preview.1"
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        pipelines = response.json().get('value', [])
        print(f"Found {len(pipelines)} total pipelines.")
        return pipelines
    except requests.exceptions.RequestException as e:
        print(f"ERROR fetching pipelines: {e}")
        return []

def evaluate_single_pipeline(pipeline_summary: Dict, token: str, branch: str, is_pr_build: bool) -> Optional[EvaluationResult]:
    """Fetches details for and evaluates a single pipeline against all criteria."""
    pipeline_name = pipeline_summary.get('name', 'Unknown')
    pipeline_id = pipeline_summary.get('id')
    print(f"\nEvaluating pipeline: '{pipeline_name}' (ID: {pipeline_id})")
    
    headers = {"Authorization": f"Bearer {token}"}
    thirty_days_ago = datetime.utcnow() - timedelta(days=DAYS_SINCE_LAST_RUN)
    target_repo_path = urlparse(TARGET_REPO_URL).path.strip('/')

    try:
        detail_url = pipeline_summary.get('url')
        if not detail_url:
            print(f"  - SKIPPING: No detail URL found for '{pipeline_name}'.")
            return None
        
        detail_response = requests.get(detail_url, headers=headers)
        detail_response.raise_for_status()
        pipeline_details: PipelineDetails = detail_response.json()

        configuration = pipeline_details['configuration']
        if configuration['type'] != 'yaml':
            print(f"  - SKIPPING: Not a YAML-based pipeline (type is '{configuration['type']}').")
            return None

        repo_info = configuration['repository']
        if repo_info['fullName'] != target_repo_path:
            repo_name = repo_info['fullName']
            print(f"  - SKIPPING: Repository name '{repo_name}' does not match target '{target_repo_path}'.")
            return None

        base_detail_url = detail_url.split('?')[0]
        runs_url = f"{base_detail_url}/runs?$top=10&api-version=7.1-preview.1"
        runs_response = requests.get(runs_url, headers=headers)
        runs_response.raise_for_status()
        runs = runs_response.json().get('value', [])

        if not runs:
            print(f"  - SKIPPING: Pipeline has no previous runs.")
            return None
        
        last_run_time_str = next((run.get('finishedDate') for run in runs if run.get('finishedDate')), None)

        if not last_run_time_str:
            print(f"  - SKIPPING: No completed runs found in the last 10 attempts.")
            return None

        if '.' in last_run_time_str:
            main_part, fractional_part = last_run_time_str.split('.', 1)
            if fractional_part.endswith('Z'):
                fractional_seconds = fractional_part[:-1]
                if len(fractional_seconds) > 6:
                    fractional_seconds = fractional_seconds[:6]
                last_run_time_str = f"{main_part}.{fractional_seconds}Z"

        last_run_time = datetime.strptime(last_run_time_str, "%Y-%m-%dT%H:%M:%S.%fZ")
        if last_run_time <= thirty_days_ago:
            print(f"  - SKIPPING: Last run was on {last_run_time.date()}, which is older than {DAYS_SINCE_LAST_RUN} days.")
            return None

        packaging_type: Literal["nightly", "release", "none"] = "none"
        if not is_pr_build:
            print(f"  - Checking YAML on branch '{branch}' for template and resource triggers...")
            yaml_path = configuration['path']
            yaml_url = f"https://raw.githubusercontent.com/{repo_info['fullName']}/{branch}/{yaml_path}"
            
            yaml_response = requests.get(yaml_url)
            yaml_response.raise_for_status()
            yaml_content = yaml_response.text

            data = yaml.safe_load(yaml_content)
            if not isinstance(data, dict):
                print("  - WARNING: YAML content is not a dictionary. Skipping checks.")
                return None

            extends_block = data.get('extends', {})
            template_path = extends_block.get('template')
            expected_template = 'v1/1ES.Official.PipelineTemplate.yml@1esPipelines'
            
            if template_path != expected_template:
                print(f"  - SKIPPING: YAML does not extend from the required template. Found: '{template_path}'.")
                return None
            print("  - OK: YAML extends from the correct template.")

            if 'resources' in data and isinstance(data.get('resources'), dict) and 'pipelines' in data.get('resources', {}):
                pipeline_list = data['resources'].get('pipelines', [])
                if isinstance(pipeline_list, list):
                    for p_resource in pipeline_list:
                        if isinstance(p_resource, dict) and 'trigger' in p_resource:
                            print(f"  - SKIPPING: YAML contains a pipeline resource with a 'trigger' key.")
                            return None
            print("  - OK: No active pipeline resource trigger found in YAML.")
            
            # Check for packaging pipeline variables/parameters
            packaging_exceptions = ['onnxruntime-ios-packaging-pipeline', '1ES-onnxruntime-Nuget-WindowsAI-Pipeline-Official']
            if 'packaging' in pipeline_name.lower() or 'nuget' in pipeline_name.lower():
                print("  - Detected packaging pipeline. Checking for required variables/parameters...")
                if pipeline_name in packaging_exceptions:
                    print(f"  - OK: Allowing exception for '{pipeline_name}' which has no build mode flags.")
                    packaging_type = "none"
                elif 'NIGHTLY_BUILD' in configuration.get('variables', {}):
                    packaging_type = "nightly"
                    print("  - OK: Found 'NIGHTLY_BUILD' pipeline variable.")
                elif any(p.get('name') == 'IsReleaseBuild' for p in data.get('parameters', [])):
                    packaging_type = "release"
                    print("  - OK: Found 'IsReleaseBuild' YAML parameter.")
                else:
                    print(f"  - SKIPPING: Packaging pipeline '{pipeline_name}' has neither a 'NIGHTLY_BUILD' variable nor an 'IsReleaseBuild' parameter.")
                    return None

        print(f"  - MATCH: '{pipeline_name}' matches all criteria.")
        return {"pipeline": pipeline_details, "packaging_type": packaging_type}

    except KeyError as e:
        print(f"  - SKIPPING '{pipeline_name}': Missing expected key {e} in pipeline details.")
    except requests.exceptions.RequestException as e:
        print(f"  - ERROR processing '{pipeline_name}': {e}")
    except yaml.YAMLError as e:
        print(f"  - WARNING for '{pipeline_name}': Could not parse YAML file: {e}. Skipping.")
    except Exception as e:
        print(f"  - ERROR processing '{pipeline_name}': An unexpected error occurred: {e}")
    
    return None

def filter_pipelines(pipelines: List[Dict], token: str, branch: str, is_pr_build: bool) -> List[EvaluationResult]:
    """Filters pipelines based on specified criteria by processing them in parallel."""
    print("\n--- Filtering Pipelines in Parallel ---")
    filtered_results: List[EvaluationResult] = []
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_pipeline = {executor.submit(evaluate_single_pipeline, p, token, branch, is_pr_build): p for p in pipelines}
        for future in as_completed(future_to_pipeline):
            result = future.result()
            if result:
                filtered_results.append(result)

    print(f"\nFound {len(filtered_results)} pipelines to trigger.")
    return filtered_results

def cancel_running_builds(pipeline_id: int, branch: str, token: str, project: str):
    """Finds and cancels any running builds for a given pipeline and branch."""
    print(f"\n--- Checking for running builds for Pipeline ID: {pipeline_id} on branch '{branch}' in project '{project}' ---")
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    
    # The 'builds' API uses 'branchName' to filter. Based on the swagger and behavior, it expects the full ref name.
    builds_url = f"https://dev.azure.com/{ADO_ORGANIZATION}/{project}/_apis/build/builds?definitions={pipeline_id}&branchName={branch}&statusFilter=inProgress,notStarted&api-version=7.1"
    
    try:
        response = requests.get(builds_url, headers=headers)
        response.raise_for_status()
        running_builds = response.json().get('value', [])

        if not running_builds:
            print("No running builds found to cancel.")
            return

        for build in running_builds:
            build_id = build['id']
            print(f"Cancelling running build ID: {build_id}...")
            cancel_url = f"https://dev.azure.com/{ADO_ORGANIZATION}/{project}/_apis/build/builds/{build_id}?api-version=7.1"
            payload = {"status": "cancelling"}
            
            patch_response = requests.patch(cancel_url, headers=headers, data=json.dumps(payload))
            patch_response.raise_for_status()
            print(f"Successfully requested cancellation for build ID: {build_id}.")

    except requests.exceptions.RequestException as e:
        print(f"ERROR checking or cancelling running builds: {e}")

def trigger_pipeline(pipeline_id: int, token: str, branch: str, project: str, nightly_override: Optional[str], release_override: Optional[str], packaging_type: str) -> Optional[int]:
    """Triggers a pipeline and returns the new build ID."""
    print(f"\n--- Triggering Pipeline ID: {pipeline_id} on branch '{branch}' in project '{project}' ---")
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    run_url = f"https://dev.azure.com/{ADO_ORGANIZATION}/{project}/_apis/pipelines/{pipeline_id}/runs?api-version=7.1-preview.1"
    
    payload: Dict[str, any] = {"resources": {"repositories": {"self": {"refName": branch}}}}

    if nightly_override is not None and packaging_type == "nightly":
        print(f"Overriding NIGHTLY_BUILD variable to '{nightly_override}'.")
        payload["variables"] = {"NIGHTLY_BUILD": {"value": nightly_override}}
    elif release_override is not None and packaging_type == "release":
        print(f"Overriding IsReleaseBuild parameter to '{release_override}'.")
        payload["templateParameters"] = {"IsReleaseBuild": release_override}

    try:
        response = requests.post(run_url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        build_info = response.json()
        build_id = build_info.get('id')
        print(f"Successfully triggered build. Build ID: {build_id}")
        print(f"   Web URL: {build_info.get('_links', {}).get('web', {}).get('href')}")
        return build_id
    except requests.exceptions.RequestException as e:
        print(f"ERROR triggering pipeline: {e}")
        if e.response is not None:
            print(f"   Response: {e.response.text}")
        return None

def main():
    """Main function to orchestrate the pipeline triggering process."""
    parser = argparse.ArgumentParser(description="Trigger Azure DevOps pipelines based on specific criteria.")
    parser.add_argument("--dry-run", action="store_true", help="Print pipelines that would be triggered, but don't actually trigger them.")
    parser.add_argument("--branch", default="main", help="The target branch for CI builds. Defaults to 'main'.")
    parser.add_argument("--pr", type=int, help="The pull request ID to trigger PR builds for. This overrides --branch.")
    parser.add_argument("--build-mode", choices=['nightly', 'release'], help="Specify the build mode for packaging pipelines (nightly or release). This sets NIGHTLY_BUILD and IsReleaseBuild accordingly.")
    args = parser.parse_args()

    project = "Lotus"
    branch_for_trigger = f"refs/heads/{args.branch}"
    branch_for_yaml_fetch = args.branch
    is_pr_build = False

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
        if args.build_mode == 'nightly':
            nightly_override = '1'
            release_override = 'false'
        elif args.build_mode == 'release':
            nightly_override = '0'
            release_override = 'true'

        for result in pipelines_to_trigger:
            pipeline = result['pipeline']
            packaging_type = result['packaging_type']
            cancel_running_builds(pipeline['id'], branch_for_trigger, token, project)
            trigger_pipeline(pipeline['id'], token, branch_for_trigger, project, nightly_override, release_override, packaging_type)

if __name__ == "__main__":
    main()