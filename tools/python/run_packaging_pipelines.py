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

def filter_pipelines(pipelines: List[Dict], token: str, branch: str, is_pr_build: bool) -> List[PipelineDetails]:
    """Filters pipelines based on specified criteria."""
    print("\n--- Filtering Pipelines ---")
    filtered_pipelines: List[PipelineDetails] = []
    headers = {"Authorization": f"Bearer {token}"}
    thirty_days_ago = datetime.utcnow() - timedelta(days=DAYS_SINCE_LAST_RUN)
    
    target_repo_path = urlparse(TARGET_REPO_URL).path.strip('/')

    for i, pipeline_summary in enumerate(pipelines):
        pipeline_name = pipeline_summary.get('name', 'Unknown')
        pipeline_id = pipeline_summary.get('id')
        print(f"\n[{i+1}/{len(pipelines)}] Evaluating pipeline: '{pipeline_name}' (ID: {pipeline_id})")
        try:
            detail_url = pipeline_summary.get('url')
            if not detail_url:
                print(f"  - SKIPPING: No detail URL found for this pipeline.")
                continue
            
            detail_response = requests.get(detail_url, headers=headers)
            detail_response.raise_for_status()
            pipeline_details: PipelineDetails = detail_response.json()

            configuration = pipeline_details['configuration']
            if configuration['type'] != 'yaml':
                print(f"  - SKIPPING: Not a YAML-based pipeline (type is '{configuration['type']}').")
                continue

            repo_info = configuration['repository']
            if repo_info['fullName'] != target_repo_path:
                repo_name = repo_info['fullName']
                print(f"  - SKIPPING: Repository name '{repo_name}' does not match target '{target_repo_path}'.")
                continue

            base_detail_url = detail_url.split('?')[0]
            runs_url = f"{base_detail_url}/runs?$top=10&api-version=7.1-preview.1"
            runs_response = requests.get(runs_url, headers=headers)
            runs_response.raise_for_status()
            runs = runs_response.json().get('value', [])

            if not runs:
                print(f"  - SKIPPING: Pipeline has no previous runs.")
                continue
            
            last_run_time_str = None
            for run in runs:
                if run.get('finishedDate'):
                    last_run_time_str = run.get('finishedDate')
                    break 

            if not last_run_time_str:
                print(f"  - SKIPPING: No completed runs found in the last 10 attempts.")
                continue

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
                continue

            # For CI builds, perform additional checks on the YAML content.
            if not is_pr_build:
                print(f"  - Checking YAML on branch '{branch}' for template and resource triggers...")
                yaml_path = configuration['path']
                yaml_url = f"https://raw.githubusercontent.com/{repo_info['fullName']}/{branch}/{yaml_path}"
                
                try:
                    yaml_response = requests.get(yaml_url)
                    yaml_response.raise_for_status()
                    yaml_content = yaml_response.text

                    try:
                        data = yaml.safe_load(yaml_content)
                        if not isinstance(data, dict):
                            print("  - WARNING: YAML content is not a dictionary. Skipping checks.")
                            continue

                        extends_block = data.get('extends', {})
                        template_path = extends_block.get('template')
                        expected_template = 'v1/1ES.Official.PipelineTemplate.yml@1esPipelines'
                        
                        if template_path != expected_template:
                            print(f"  - SKIPPING: YAML does not extend from the required template. Found: '{template_path}'.")
                            continue
                        print("  - OK: YAML extends from the correct template.")

                        is_triggered_by_pipeline = False
                        if 'resources' in data and isinstance(data.get('resources'), dict) and 'pipelines' in data.get('resources', {}):
                            pipeline_list = data['resources'].get('pipelines', [])
                            if isinstance(pipeline_list, list):
                                for p_resource in pipeline_list:
                                    if isinstance(p_resource, dict) and 'trigger' in p_resource:
                                        print(f"  - SKIPPING: YAML contains a pipeline resource with a 'trigger' key.")
                                        is_triggered_by_pipeline = True
                                        break
                        
                        if is_triggered_by_pipeline:
                            continue
                        print("  - OK: No active pipeline resource trigger found in YAML.")

                    except yaml.YAMLError as e:
                        print(f"  - WARNING: Could not parse YAML file: {e}. Skipping checks.")
                        continue

                except requests.exceptions.RequestException as e:
                    print(f"  - WARNING: Could not fetch YAML file from {yaml_url}. Error: {e}. Skipping checks.")
                    continue
            
            print(f"  - MATCH: '{pipeline_name}' matches all criteria.")
            filtered_pipelines.append(pipeline_details)

        except KeyError as e:
            print(f"  - SKIPPING: Missing expected key {e} in pipeline details.")
            continue
        except requests.exceptions.RequestException as e:
            print(f"  - ERROR: Could not process pipeline '{pipeline_name}'. Error: {e}")
            if e.response:
                print(f"    Response Status: {e.response.status_code}, Body: {e.response.text[:200]}...")
        except Exception as e:
            print(f"  - ERROR: An unexpected error occurred while processing pipeline '{pipeline_name}': {e}")

    print(f"\nFound {len(filtered_pipelines)} pipelines to trigger.")
    return filtered_pipelines

def cancel_running_builds(pipeline_id: int, branch: str, token: str, project: str):
    """Finds and cancels any running builds for a given pipeline and branch."""
    print(f"\n--- Checking for running builds for Pipeline ID: {pipeline_id} on branch '{branch}' in project '{project}' ---")
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    
    # The 'builds' API uses 'branchName' which expects the short name (e.g., 'main' or 'pull/12345/head')
    short_branch_name = branch.replace('refs/heads/', '').replace('refs/', '')
    builds_url = f"https://dev.azure.com/{ADO_ORGANIZATION}/{project}/_apis/build/builds?definitions={pipeline_id}&branchName={short_branch_name}&statusFilter=inProgress,notStarted&api-version=7.1"
    
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

def trigger_pipeline(pipeline_id: int, token: str, branch: str, project: str) -> Optional[int]:
    """Triggers a pipeline and returns the new build ID."""
    print(f"\n--- Triggering Pipeline ID: {pipeline_id} on branch '{branch}' in project '{project}' ---")
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    run_url = f"https://dev.azure.com/{ADO_ORGANIZATION}/{project}/_apis/pipelines/{pipeline_id}/runs?api-version=7.1-preview.1"
    payload = {"resources": {"repositories": {"self": {"refName": branch}}}}

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
    args = parser.parse_args()

    project = "Lotus"
    branch = f"refs/heads/{args.branch}"
    is_pr_build = False

    if args.pr:
        print(f"--- Pull Request Mode Activated for PR #{args.pr} ---")
        project = "PublicPackages"
        branch = f"refs/pull/{args.pr}/head"
        is_pr_build = True
    else:
        print(f"--- CI Build Mode Activated for branch '{args.branch}' ---")


    if args.dry_run:
        print("DRY RUN MODE ENABLED: No pipelines will be triggered.")

    token = get_azure_cli_token()
    all_pipelines = get_pipelines(token, project)
    if not all_pipelines:
        return

    pipelines_to_trigger = filter_pipelines(all_pipelines, token, args.branch, is_pr_build)

    if not pipelines_to_trigger:
        print("\nNo pipelines matching the criteria were found.")
        return

    if args.dry_run:
        print(f"\n--- The following pipelines would be triggered on branch '{branch}' ---")
        for pipeline in pipelines_to_trigger:
            print(f"  - {pipeline['name']} (ID: {pipeline['id']})")
    else:
        print(f"\n--- Triggering {len(pipelines_to_trigger)} Pipelines on branch '{branch}' ---")
        for pipeline in pipelines_to_trigger:
            cancel_running_builds(pipeline['id'], branch, token, project)
            trigger_pipeline(pipeline['id'], token, branch, project)

if __name__ == "__main__":
    main()