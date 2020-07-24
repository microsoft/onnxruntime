#!/usr/bin/python3
import urllib.request
import json
import subprocess
import os
import sys
import shutil
import argparse
from urllib.parse import urlparse
from urllib.parse import urljoin
from urllib.parse import urlsplit

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
REPO_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.append(os.path.join(REPO_DIR, "tools", "python"))

from get_azcopy import get_azcopy  # noqa: E402


# Hardcoded map of storage account to azure region endpoint
storage_account_to_endpoint_map = {
    'onnxruntimetestdata.blob.core.windows.net': {
        'centralus': 'onnxruntimetestdata.blob.core.windows.net',
        'northcentralus': 'onnxruntimetestdata2.blob.core.windows.net'
    },
    'ortinternaltestdata': {
        'centralus': 'ortinternaltestdata.blob.core.windows.net'
    }
}


def is_windows():
    return sys.platform.startswith("win")


def get_azure_region():
    # Try to get azure location for this machine
    if args.azure_region is not None:
        return args.azure_region
    else:
        azure_region = None
        try:
            req = urllib.request.Request('http://169.254.169.254/metadata/instance?api-version=2018-10-01')
            req.add_header('Metadata', 'true')
            body = urllib.request.urlopen(req).read()
            body = json.loads(body.decode('utf-8'))
            azure_region = body['compute']['location']
            print("This VM is in azure region: %s" % azure_region)
        except Exception as e:
            print(e)
            print("Failed to get azure region for this machine.")
    return azure_region


def parse_arguments():
    parser = argparse.ArgumentParser(description="ONNXRuntime Data Downloader.")
    parser.add_argument("--test_data_url", required=True, help="Test data URL.")
    parser.add_argument("--azure_region", help="Azure region")
    parser.add_argument("--build_dir", required=True, help="Path to the build directory.")
    parser.add_argument("--edge_device", action="store_true", help="Edge device with limit disk space.")
    return parser.parse_args()


def get_endpoint_by_region(current_endpoint, azure_location):
    # Find storage account closest to the current region
    # TODO : Ideally we should replicate same storage account to multiple regions
    # and choose the endpoint accordigly instead of the current way where
    # we maintain separate storage accounts per region.
    if current_endpoint in storage_account_to_endpoint_map:
        endpoint_to_region_map = storage_account_to_endpoint_map[current_endpoint]
        if azure_location in endpoint_to_region_map:
            return endpoint_to_region_map[azure_location]
        else:
            print('warning: no local data cache for azure region %s' % azure_location)
            return current_endpoint
    else:
        print('warning: no local data cache for azure region %s' % azure_location)
        return current_endpoint
    return current_endpoint


def get_region_based_url(url, azure_location):
    current_endpoint = urlsplit(url).netloc
    endpoint_by_region = get_endpoint_by_region(current_endpoint, azure_location)
    url = url.replace(current_endpoint, endpoint_by_region)
    print("url changed to %s" % url)
    return url


def download_and_unzip(azcopy_path, build_dir, url, dest_folder, use_token=True):
    dest_folder = os.path.join(build_dir, dest_folder)
    # attach the SAS token to the url. Note DO NOT print the url with the token in any logs.
    token = os.environ.get('Test_Data_Download_Key')
    if use_token:
        url_with_token = urljoin(url, token)
    else:
        url_with_token = url

    # Download data using AZCopy tool
    try:
        subprocess.run(
            [azcopy_path, 'cp', '--log-level', 'ERROR', '--recursive', url_with_token, build_dir],
            check=True)
    except Exception as e:
        print(e)
        print(azcopy_path)
        raise Exception("Downloading data failed. Source: " + url + " Destination: " + build_dir)

    os.makedirs(dest_folder, exist_ok=True)
    local_file_name = os.path.join(build_dir, os.path.basename(urlparse(url).path))
    if is_windows():
        print("unzip %s" % local_file_name)
        if shutil.which('7z'):  # 7-Zip
            subprocess.run(['7z', 'x', local_file_name, '-y', '-o' + dest_folder], check=True)
        elif shutil.which('7za'):  # 7-Zip standalone
            subprocess.run(['7za', 'x', local_file_name, '-y', '-o' + dest_folder], check=True)
        else:
            print("No unzip tool for use")
            sys.exit(1)
    else:
        subprocess.run(['unzip', '-qd', dest_folder, local_file_name], check=True)
    os.unlink(local_file_name)


args = parse_arguments()
models_folder = 'models'

if args.edge_device:
    dest_folder = os.path.join(args.build_dir, models_folder)
    # For edge device, the model zip file is persist at /mnt/ubuntu/tmp/model.zip
    local_file_name = '/mnt/ubuntu/tmp/model.zip'
    if os.path.exists(local_file_name):
        subprocess.run(['unzip', '-qd', dest_folder, local_file_name], check=True)
    else:
        raise Exception(local_file_name + " does not exist on edge device. Downloading test data step failed.")
else:
    all_downloads_done = False
    azure_region = args.azure_region
    if not azure_region:
        azure_region = get_azure_region()
    try:
        # Download test data
        with get_azcopy(os.path.join(args.build_dir, "azcopy")) as azcopy_path:
            url = get_region_based_url(args.test_data_url, azure_region)
            print("Starting test data download %s" % url)
            download_and_unzip(azcopy_path, args.build_dir, url, models_folder)

        all_downloads_done = True

    except Exception as e:
        print(e)

    if not all_downloads_done:
        raise Exception("Downloading test data step failed.")
