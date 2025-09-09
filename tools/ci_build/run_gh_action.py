# scripts/run_action.py
import os
import subprocess
import requests
import zipfile
import shutil
import glob
import sys
import re

def run():
    # --- 1. Configuration ---
    if len(sys.argv) < 2:
        print("::error::Action version argument was not provided.")
        sys.exit(1)
        
    action_version = sys.argv[1]
    
    # NOTE: Replace placeholder hashes with actual, valid SHA512 hashes for the chosen versions and platform.
    action_inputs = {
        'INPUT_CMAKE-VERSION': '3.29.0',
        'INPUT_CMAKE-HASH': '8482e754bf5bf45349ba2f2184999f81f8754ed3d281e1708f1f9a3b2fcd05c3aa5368e6247930495722ffc5982aadbe489630c5716241ab1702c3cf866483cf',
        'INPUT_VCPKG-VERSION': '2023.10.19',
        'INPUT_VCPKG-HASH': '17e96169cd3f266c4716fcdc1bb728e6a64f103941ece463a2834d50694eba4fb48f30135503fd466402afa139abc847ef630733c442595d1c34979f261b0114',
        'INPUT_ADD-CMAKE-TO-PATH': 'true',
        'INPUT_DISABLE-TERRAPIN': 'true'
    }

    # --- 2. Download and Extract the Action ---
    zip_url = f"https://github.com/microsoft/onnxruntime-github-actions/archive/refs/tags/{action_version}.zip"
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

    print(f"Running action script and saving output to {output_log_path}...")
    with open(output_log_path, 'w') as log_file:
        process = subprocess.run(
            ['node', action_script_path],
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True
        )

    # --- 4. Process the Action's Output Log ---
    print("\n--- Processing Action Output ---")
    if process.returncode != 0:
        print(f"##vso[task.logissue type=error]Action script failed with exit code {process.returncode}.")
        with open(output_log_path, 'r') as log_file:
            print(log_file.read())
        sys.exit(1)

    with open(output_log_path, 'r') as log_file:
        for line in log_file:
            line = line.strip()
            # Handle setting environment variables
            if line.startswith('::set-env name='):
                match = re.match(r'::set-env name=([^:]+)::(.*)', line)
                if match:
                    var_name, var_value = match.groups()
                    print(f"Found set-env command for '{var_name}'. Emitting VSO command.")
                    # Emitting the ADO command to set a variable for subsequent steps
                    print(f"##vso[task.setvariable variable={var_name}]{var_value}")
            
            # Handle adding to PATH
            elif line.startswith('::add-path::'):
                path_to_add = line[len('::add-path::'):]
                print(f"Found add-path command for '{path_to_add}'. Emitting VSO command.")
                # Emitting the ADO command to prepend to the PATH
                print(f"##vso[task.prependpath]{path_to_add}")

    print("--- Finished Processing Action Output ---")

if __name__ == "__main__":
    run()