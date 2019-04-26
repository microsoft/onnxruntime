#!/usr/bin/python3

import urllib.request
import json
import subprocess
import os
import argparse
from urllib.parse import urlparse

def get_azure_region():
    req = urllib.request.Request('http://169.254.169.254/metadata/instance?api-version=2018-10-01')
    req.add_header('Metadata', 'true')
    body = urllib.request.urlopen(req).read()
    body = json.loads(body.decode('utf-8'))
    return body['compute']['location']

def parse_arguments():
    parser = argparse.ArgumentParser(description="ONNXRuntime Data Downloader.")
    parser.add_argument("--test_data_url", help="Test data URL.")
    return parser.parse_args()


def get_server_hostname():
    #should be northcentralus or centralus
    azure_location=get_azure_region()
    print(azure_location)
    if azure_location == 'centralus':
        hostname='onnxruntimetestdata'
    elif azure_location == 'northcentralus':
        hostname='onnxruntimetestdata2'
    else:
        print('warning: no local data cache for azure region %s' % azure_location)
        hostname='onnxruntimetestdata2'
    return hostname

args = parse_arguments()
hostname=get_server_hostname()
url=args.test_data_url.replace('onnxruntimetestdata', hostname)
print('data url=%s' % url)
subprocess.run(['./azcopy','cp', '--log-level','ERROR', url,'.'],check=True)
os.makedirs('models',exist_ok=True)
local_file_name = os.path.basename(urlparse(url).path)
subprocess.run(['unzip', '-qd','models',local_file_name])
