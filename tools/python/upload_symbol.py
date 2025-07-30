#!/usr/bin/env python3
# See PythonTools.md for how to use it.

import argparse
import concurrent.futures
import fnmatch
import json
import os
import platform
import shutil
import subprocess
import sys
import time # For basic timing
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import requests  # type: ignore # Ignore type checking for requests if no stubs installed

# --- Configuration ---
GITHUB_API_URL = "https://api.github.com"
REPO_OWNER = "microsoft"
REPO_NAME = "onnxruntime"
ASSET_PATTERN = "onnxruntime-win-*.zip"
SYMBOL_EXE = "Symbol.exe"  # Assume it's in PATH, or provide the full path
SYMBOL_SERVER_BASE_URL = "https://{org_name}.artifacts.visualstudio.com/"
SYMBOL_EXPIRATION_DAYS = 1095
DOWNLOAD_CHUNK_SIZE = 8192 # bytes for download progress
DEFAULT_MAX_WORKERS = 4 # Adjust parallelism level as needed

# --- Type Definitions for Clarity (Requires Python 3.8+) ---
# Using TypedDict, requires Python 3.8+ (already covered by 3.12+ requirement)
class AssetInfo(TypedDict):
    name: str
    browser_download_url: str
    # Add other relevant fields if needed

class ProcessingResult(TypedDict):
    asset_name: str
    success: bool
    zip_path: Optional[Path]
    extract_path: Optional[Path]
    message: str

def fetch_release_data(version: str, github_token: Optional[str]) -> Optional[Dict[str, Any]]:
    """Fetches release data for a specific tag from the GitHub API, using token if provided."""
    tag_name = f"v{version}"
    url = f"{GITHUB_API_URL}/repos/{REPO_OWNER}/{REPO_NAME}/releases/tags/{tag_name}"
    print(f"Fetching release data from: {url}") # Keep initial fetch log outside parallel tasks
    try:
        headers = {}
        if github_token:
            headers["Authorization"] = f"token {github_token}"
            # Consider adding User-Agent
            # headers["User-Agent"] = "YourAppNameOrScript"
        response = requests.get(url, timeout=60, headers=headers) # Increased timeout slightly
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching release data for tag {tag_name}: {e}", file=sys.stderr)
        if isinstance(e, requests.exceptions.HTTPError) and e.response is not None:
            if e.response.status_code == 404:
                print(f"Release tag '{tag_name}' not found.", file=sys.stderr)
            elif e.response.status_code == 403:
                 print(f"Access forbidden (403). Check your GitHub token permissions or API rate limits.", file=sys.stderr)
            elif e.response.status_code == 401:
                 print(f"Authentication failed (401). Check your GitHub token.", file=sys.stderr)
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response from GitHub API: {e}", file=sys.stderr)
        return None

def download_asset(asset_url: str, download_path: Path, asset_name: str) -> bool:
    """Downloads an asset from a URL to a specified path."""
    # Use asset_name for better logging in parallel context
    print(f"[{asset_name}] Downloading: {asset_url}\n  To: {download_path}")
    try:
        # Ensure parent directory exists (important if download_path includes subdirs)
        download_path.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(asset_url, stream=True, timeout=900) as response: # 15 min timeout
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0
            last_print_time = time.time()
            with open(download_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    # Throttle progress printing in parallel to avoid spam
                    current_time = time.time()
                    if total_size > 0 and (current_time - last_print_time > 2 or downloaded_size == total_size) :
                        percent = int(100 * downloaded_size / total_size)
                        print(f"\r[{asset_name}] Progress: {downloaded_size / (1024*1024):.2f} MB / {total_size / (1024*1024):.2f} MB ({percent}%)", end="")
                        last_print_time = current_time
            print(f"\n[{asset_name}] Download complete.")
            return True
    except requests.exceptions.RequestException as e:
        print(f"\n[{asset_name}] Error downloading {asset_url}: {e}", file=sys.stderr)
        if download_path.exists():
            try:
                download_path.unlink()
            except OSError as unlink_err:
                 print(f"[{asset_name}] Warning: Could not remove incomplete download {download_path}: {unlink_err}", file=sys.stderr)
        return False
    except IOError as e:
        print(f"\n[{asset_name}] Error writing file {download_path}: {e}", file=sys.stderr)
        return False

def unzip_asset(zip_path: Path, extract_dir: Path, asset_name: str) -> bool:
    """
    Unzips a file into the specified dedicated directory, stripping the first level
    directory component from member paths (like tar --strip-components=1).
    """
    print(f"[{asset_name}] Unzipping: {zip_path}\n  To: {extract_dir} (stripping 1 level)")
    try:
        # Ensure the dedicated extraction directory exists
        extract_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            members_to_extract = zip_ref.infolist()
            if not members_to_extract:
                print(f"[{asset_name}] Warning: Zip file is empty.")
                return True # Not an error

            extracted_count = 0
            for member in members_to_extract:
                original_path = Path(member.filename)
                parts = original_path.parts

                if len(parts) > 1:
                    stripped_rel_path = Path(*parts[1:])
                    # Target path is within the dedicated extract_dir
                    target_path = extract_dir / stripped_rel_path

                    if member.is_dir():
                        target_path.mkdir(parents=True, exist_ok=True)
                    else:
                        target_path.parent.mkdir(parents=True, exist_ok=True)
                        try:
                            with zip_ref.open(member, 'r') as source, open(target_path, 'wb') as target:
                                shutil.copyfileobj(source, target)
                            extracted_count += 1
                        except Exception as copy_exc:
                             print(f"\n[{asset_name}] Error extracting file {member.filename} to {target_path}: {copy_exc}", file=sys.stderr)
                             # Decide if this is fatal for the whole unzip operation
                             # return False # Option: Fail fast
                             continue # Option: Try to extract other files
                # else:
                    # print(f"[{asset_name}] Skipping root item: {member.filename}")

        print(f"[{asset_name}] Unzip complete. Extracted {extracted_count} files.")
        return True
    except zipfile.BadZipFile:
        print(f"[{asset_name}] Error: {zip_path} is not a valid zip file or is corrupted.", file=sys.stderr)
        return False
    except Exception as e:
        print(f"[{asset_name}] Error unzipping {zip_path} to {extract_dir}: {e}", file=sys.stderr)
        return False

def publish_symbols(org_name: str, lib_dir: Path, package_name: str, asset_name: str) -> bool:
    """Runs the Symbol.exe publish command."""
    symbol_server_url = SYMBOL_SERVER_BASE_URL.format(org_name=org_name)
    command: List[str] = [
        SYMBOL_EXE,
        "publish",
        "-s", symbol_server_url,
        "--expirationInDays", str(SYMBOL_EXPIRATION_DAYS),
        "-d", str(lib_dir),
        "-n", package_name,
    ]
    print(f"[{asset_name}] Running symbol publish command for: {package_name}")
    print(f"[{asset_name}]   Command: {' '.join(command)}")

    try:
        use_shell = platform.system() == "Windows" and SYMBOL_EXE.lower().endswith((".bat", ".cmd"))
        # Increased timeout for symbol publishing, can take time
        result = subprocess.run(command, check=True, capture_output=True, text=True, shell=use_shell, timeout=1800) # 30 min timeout
        print(f"[{asset_name}] Symbol publish command successful for {package_name}.")
        # Only print stdout/stderr if they contain something useful (optional)
        if result.stdout and result.stdout.strip():
             print(f"[{asset_name}]   Stdout: {result.stdout.strip()}")
        if result.stderr and result.stderr.strip():
             print(f"[{asset_name}]   Stderr: {result.stderr.strip()}", file=sys.stderr)
        return True
    except FileNotFoundError:
        print(f"[{asset_name}] Error: '{SYMBOL_EXE}' command not found. Make sure it's in your PATH or provide the full path.", file=sys.stderr)
        return False
    except subprocess.TimeoutExpired:
        print(f"[{asset_name}] Error: Symbol.exe command timed out after 30 minutes for {package_name}.", file=sys.stderr)
        return False
    except subprocess.CalledProcessError as e:
        print(f"[{asset_name}] Error running Symbol.exe for {package_name}:", file=sys.stderr)
        print(f"[{asset_name}]   Return code: {e.returncode}", file=sys.stderr)
        print(f"[{asset_name}]   Command: {' '.join(e.cmd)}", file=sys.stderr)
        if e.stdout and e.stdout.strip(): print(f"[{asset_name}]   Stdout: {e.stdout.strip()}", file=sys.stderr)
        if e.stderr and e.stderr.strip(): print(f"[{asset_name}]   Stderr: {e.stderr.strip()}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"[{asset_name}] An unexpected error occurred while running Symbol.exe: {e}", file=sys.stderr)
        return False

# --- Asset Processing Pipeline Function ---

def process_asset_pipeline(
    asset: AssetInfo,
    org_name: str,
    output_base_dir: Path
) -> ProcessingResult:
    """Downloads, unzips (with stripping), and publishes symbols for a single asset."""
    asset_name = asset['name']
    asset_url = asset['browser_download_url']
    zip_filename = output_base_dir / asset_name
    # Create a unique subdirectory for extraction based on the zip filename stem
    extract_dir_name = Path(asset_name).stem
    extract_path = output_base_dir / extract_dir_name

    print(f"[{asset_name}] Starting processing pipeline...")

    # 1. Download
    if not download_asset(asset_url, zip_filename, asset_name):
        return ProcessingResult(asset_name=asset_name, success=False, zip_path=zip_filename, extract_path=None, message="Download failed")

    # 2. Unzip (into dedicated directory, stripping 1 level inside it)
    if not unzip_asset(zip_filename, extract_path, asset_name):
        return ProcessingResult(asset_name=asset_name, success=False, zip_path=zip_filename, extract_path=extract_path, message="Unzip failed")

    # 3. Check for 'lib' directory inside the dedicated, stripped extraction path
    lib_dir = extract_path / "lib"
    if not lib_dir.is_dir():
        message = f"Expected 'lib' directory not found in {extract_path} after stripping."
        print(f"[{asset_name}] Error: {message}", file=sys.stderr)
        return ProcessingResult(asset_name=asset_name, success=False, zip_path=zip_filename, extract_path=extract_path, message=message)

    # 4. Publish Symbols
    package_name = extract_dir_name # Use the unique dir name as the package name
    if not publish_symbols(org_name, lib_dir, package_name, asset_name):
         return ProcessingResult(asset_name=asset_name, success=False, zip_path=zip_filename, extract_path=extract_path, message="Symbol publishing failed")

    print(f"[{asset_name}] Processing pipeline completed successfully.")
    return ProcessingResult(asset_name=asset_name, success=True, zip_path=zip_filename, extract_path=extract_path, message="Success")

# --- Main Execution Logic ---

def main() -> None:
    """Main script logic."""
    parser = argparse.ArgumentParser(
        description=f"Download ONNX Runtime Windows release assets, unzip (stripping 1 level into unique dirs), and publish symbols in parallel using {SYMBOL_EXE}."
    )
    # --- Arguments ---
    parser.add_argument(
        "-v", "--version",
        type=str,
        required=True, # Now a required option
        help="The ONNX Runtime version to process (e.g., 1.17.0).",
    )
    parser.add_argument(
        "-o", "--org-name", # Changed to --org-name flag (conventional)
        type=str,
        required=True, # Now a required option
        dest="org_name", # Ensure it's stored in args.org_name
        help="Your Azure DevOps organization name.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("./onnxruntime_symbols_work"),
        help="Base directory to download zips and create extraction subdirectories. Defaults to './onnxruntime_symbols_work'."
    )
    parser.add_argument(
        "--cleanup", action=argparse.BooleanOptionalAction, default=True,
        help="Delete downloaded zip files and extracted directories after processing."
    )
    parser.add_argument(
        "--github-token", type=str, default=os.environ.get("GITHUB_TOKEN"), # Read from env var if set
        help="GitHub Personal Access Token (PAT) for API requests (avoids rate limits). Can also be set via GITHUB_TOKEN env var."
    )
    parser.add_argument(
        "--max-workers", type=int, default=DEFAULT_MAX_WORKERS,
        help=f"Maximum number of parallel workers. Defaults to {DEFAULT_MAX_WORKERS}."
    )

    args = parser.parse_args()

    # Access arguments using args.version and args.org_name as before
    version: str = args.version
    org_name: str = args.org_name
    output_base_dir: Path = args.output_dir.resolve()
    perform_cleanup: bool = args.cleanup
    github_token: Optional[str] = args.github_token
    max_workers: int = args.max_workers if args.max_workers > 0 else DEFAULT_MAX_WORKERS


    start_time = time.time()
    print(f"--- Starting Symbol Publishing for ONNX Runtime v{version} (Parallel) ---")
    print(f"Azure DevOps Org: {org_name}")
    print(f"Base Working Directory: {output_base_dir}")
    print(f"Cleanup Enabled: {perform_cleanup}")
    print(f"Max Workers: {max_workers}")
    if github_token:
        print("Using GitHub Token for API requests.")
    else:
        print("No GitHub Token provided. You may hit API rate limits.")


    # 1. Fetch release information (sequentially first)
    release_data = fetch_release_data(version, github_token)
    if not release_data:
        sys.exit(1)

    # 2. Filter assets
    assets = release_data.get("assets", [])
    # Ensure assets have the required fields before casting
    valid_assets_data = [
        a for a in assets
        if isinstance(a, dict) and "name" in a and "browser_download_url" in a
           and fnmatch.fnmatch(a["name"], ASSET_PATTERN)
    ]
    win_zip_assets: List[AssetInfo] = valid_assets_data # Cast to AssetInfo list


    if not win_zip_assets:
        print(f"No assets found matching pattern '{ASSET_PATTERN}' for version {version}.", file=sys.stderr)
        sys.exit(0) # Exit cleanly if no assets match

    print(f"Found {len(win_zip_assets)} matching assets to process in parallel:")
    for asset in win_zip_assets:
        print(f"  - {asset['name']}")

    # Create base output directory if it doesn't exist
    try:
        output_base_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"Error creating base output directory {output_base_dir}: {e}", file=sys.stderr)
        sys.exit(1)

    # --- Process assets in parallel ---
    print(f"\n--- Processing {len(win_zip_assets)} assets using up to {max_workers} workers ---")
    results: List[ProcessingResult] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_asset: Dict[concurrent.futures.Future[ProcessingResult], str] = {
            executor.submit(process_asset_pipeline, asset, org_name, output_base_dir): asset['name']
            for asset in win_zip_assets
        }

        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_asset):
            asset_name = future_to_asset[future]
            try:
                result = future.result()
                results.append(result)
                status = "SUCCESS" if result['success'] else f"FAILED ({result['message']})"
                print(f"--- Result for [{asset_name}]: {status} ---")
            except Exception as exc:
                # Catch exceptions raised within the pipeline function itself
                print(f"--- Exception for [{asset_name}]: {exc} ---", file=sys.stderr)
                # Record failure, attempt to find paths if possible (might be tricky)
                results.append(ProcessingResult(asset_name=asset_name, success=False, zip_path=None, extract_path=None, message=f"Pipeline execution error: {exc}"))


    # --- Aggregate results ---
    successful_tasks = [r for r in results if r['success']]
    failed_tasks = [r for r in results if not r['success']]
    all_successful = len(failed_tasks) == 0

    print("\n--- Parallel Processing Summary ---")
    print(f"Total assets processed: {len(results)}")
    print(f"Successful: {len(successful_tasks)}")
    print(f"Failed: {len(failed_tasks)}")
    if failed_tasks:
        print("Failed assets:")
        for task in failed_tasks:
            print(f"  - {task['asset_name']}: {task['message']}")

    # --- Cleanup (Now safe with unique directories) ---
    if perform_cleanup:
        print("\n--- Cleaning up processed items ---")
        cleaned_count = 0
        for result in results:
            asset_name = result['asset_name']
            zip_path = result.get('zip_path')
            extract_path = result.get('extract_path')

            # Clean up extracted directory (if it exists)
            if extract_path and extract_path.exists() and extract_path.is_dir():
                 # Check it's within the intended base directory for safety
                if output_base_dir in extract_path.parents:
                    try:
                        print(f"[{asset_name}] Removing extracted directory: {extract_path}")
                        shutil.rmtree(extract_path)
                        cleaned_count +=1
                    except OSError as e:
                        print(f"[{asset_name}] Warning: Could not remove extracted directory {extract_path}: {e}", file=sys.stderr)
                else:
                     print(f"[{asset_name}] Warning: Skipping cleanup of {extract_path} as it seems outside the base working directory.", file=sys.stderr)


            # Clean up downloaded zip file (if it exists)
            if zip_path and zip_path.exists() and zip_path.is_file():
                try:
                    print(f"[{asset_name}] Removing zip file: {zip_path}")
                    zip_path.unlink()
                except OSError as e:
                    print(f"[{asset_name}] Warning: Could not remove zip file {zip_path}: {e}", file=sys.stderr)
        print(f"Cleanup attempted for {cleaned_count} extracted directories.")
    else:
        print("\n--- Skipping Cleanup ---")
        print(f"Downloaded zip files and extracted directories remain in: {output_base_dir}")

    # --- Final Status ---
    total_time = time.time() - start_time
    print(f"\n--- Symbol Publishing Process Finished in {total_time:.2f} seconds ---")
    if all_successful:
        print("All tasks completed successfully.")
        sys.exit(0)
    else:
        print("One or more tasks failed. Please check the output above for errors.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__": 
    main()