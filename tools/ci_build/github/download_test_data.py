#!/usr/bin/python3
import urllib.request
import json
import subprocess
import os
import sys
import shutil
import argparse
from urllib.parse import urlparse

def is_windows():
    return sys.platform.startswith("win")

def get_azure_region():
    req = urllib.request.Request('http://169.254.169.254/metadata/instance?api-version=2018-10-01')
    req.add_header('Metadata', 'true')
    body = urllib.request.urlopen(req).read()
    body = json.loads(body.decode('utf-8'))
    return body['compute']['location']

def parse_arguments():
    parser = argparse.ArgumentParser(description="ONNXRuntime Data Downloader.")
    parser.add_argument("--test_data_url", help="Test data URL.")
    parser.add_argument("--azure_region", help="Azure region")
    parser.add_argument("--build_dir", required=True, help="Path to the build directory.")
    return parser.parse_args()


def get_server_hostname(azure_location):
    if azure_location is None:
      #should be northcentralus or centralus
      azure_location = get_azure_region()
    print("This VM is in azure location: %s" % azure_location)
    if azure_location == 'centralus':
        hostname = 'onnxruntimetestdata'
    elif azure_location == 'northcentralus':
        hostname = 'onnxruntimetestdata2'
    else:
        print('warning: no local data cache for azure region %s' % azure_location)
        hostname = 'onnxruntimetestdata2'
    return hostname


def download_and_unzip(build_dir, url, dest_folder):
    subprocess.run([os.path.join(build_dir,'azcopy'),'cp', '--log-level','ERROR', url, build_dir],check=True)
    os.makedirs(dest_folder,exist_ok=True)
    local_file_name = os.path.basename(urlparse(url).path)
    if is_windows():
      if shutil.which('7z'):  # 7-Zip
          run_subprocess(['7z','x', local_file_name, '-y', '-o', dest_folder])
      elif shutil.which('7za'):  # 7-Zip standalone
          run_subprocess(['7za', 'x', local_file_name, '-y', '-o', dest_folder])
      else:
          log.error("No unzip tool for use")   
          sys.exit(1)
    else:
       subprocess.run(['unzip','-qd', dest_folder ,local_file_name], check=True)

args = parse_arguments()
hostname = get_server_hostname(args.azure_region)
url = args.test_data_url.replace('onnxruntimetestdata', hostname)
print('data url=%s' % url)
download_and_unzip(args.build_dir, url, 'models')
if is_windows():
    url = 'https://onnxruntimetestdata.blob.core.windows.net/models/cmake-3.15.1-win64-x64.zip'
    url = args.test_data_url.replace('onnxruntimetestdata', hostname)
    download_and_unzip(args.build_dir, url, 'cmake_temp')
    dest_dir = os.path.join(args.build_dir,'cmake')
    if os.path.exists(dest_dir):
        print('deleting %s' % dest_dir)
        shutil.rmtree(dest_dir)
    shutil.move(os.path.join(args.build_dir,'cmake_temp','cmake-3.13.2-win64-x64'),dest_dir)


