# scripts/run_action.py
import os
import subprocess
import requests
import zipfile
import shutil
import glob
import sys
import re
from util import run
def run():
    # --- 1. Configuration ---
    if len(sys.argv) < 2:
        print("::error::Action version argument was not provided.")
        sys.exit(1)
        
    action_version = sys.argv[1]
    
    action_inputs = {
      'INPUT_CMAKE-VERSION': '3.31.8',
      'INPUT_CMAKE-HASH': '99cc9c63ae49f21253efb5921de2ba84ce136018abf08632c92c060ba91d552e0f6acc214e9ba8123dee0cf6d1cf089ca389e321879fd9d719a60d975bcffcc8',
      'INPUT_VCPKG-VERSION': '2025.06.13',
      'INPUT_VCPKG-HASH': '735923258c5187966698f98ce0f1393b8adc6f84d44fd8829dda7db52828639331764ecf41f50c8e881e497b569f463dbd02dcb027ee9d9ede0711102de256cc',
      'INPUT_ADD-CMAKE-TO-PATH': 'true',
      'INPUT_DISABLE-TERRAPIN': 'true'
    }


    # --- 2. Download and Extract the Action ---
    # zip_url = f"https://github.com/microsoft/onnxruntime-github-actions/archive/refs/tags/{action_version}.zip"
    zip_url = "https://github.com/microsoft/onnxruntime-github-actions/archive/0e4007c285f5d78de5496953aa8696dc80f8f735.zip"
    artifacts_dir = os.environ.get('SYSTEM_ARTIFACTSDIRECTORY', '.')
    zip_path = os.path.join(artifacts_dir, "action.zip")
    extract_dir = os.path.join(artifacts_dir, "action-unzipped")
    output_log_path = 'action_output.log'

    print(f"Downloading action source from: {zip_url}")
    response = requests.get(zip_url, stream=True)
    response.raise_for_status()
    with open(zip_path, 'wb') as f:
        shutil.copyfileobj(response.raw, f)

    print(f"Extracting {zip_path} to {extract_dir}")
    if os.path.exists(extract_dir):
        shutil.rmtree(extract_dir)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    # --- 3. Locate and Run the Action Script ---
    search_pattern = os.path.join(extract_dir, 'onnxruntime-github-actions-*')
    extracted_folders = glob.glob(search_pattern)
    if not extracted_folders:
        raise Exception(f"Could not find extracted action directory matching '{search_pattern}'")
    action_base_path = extracted_folders[0]
    print(f"Found action base path: {action_base_path}")

    action_script_path = os.path.join(action_base_path, 'setup-build-tools', 'dist', 'index.js')
    if not os.path.exists(action_script_path):
        raise FileNotFoundError(f"Action script not found at expected path: {action_script_path}")

    env = os.environ.copy()
    env.update(action_inputs)

    # Map ADO environment variables to what the GitHub Action script expects
    if 'AGENT_TOOLSDIRECTORY' in env:
        env['RUNNER_TOOL_CACHE'] = env['AGENT_TOOLSDIRECTORY']
        print(f"Mapped RUNNER_TOOL_CACHE to AGENT_TOOLSDIRECTORY: {env['RUNNER_TOOL_CACHE']}")
    if 'AGENT_TEMPDIRECTORY' in env:
        env['RUNNER_TEMP'] = env['AGENT_TEMPDIRECTORY']
        print(f"Mapped RUNNER_TEMP to AGENT_TEMPDIRECTORY: {env['RUNNER_TEMP']}")


    process = run(['node', action_script_path], env=env)

if __name__ == "__main__":
    run()