import argparse
import azure.devops.connection
import msrest.authentication
import getpass
import re
import requests
import shutil
import subprocess
import zipfile

from pathlib import Path
from requests.auth import HTTPBasicAuth


def parse_args():
    class Formatter(argparse.RawTextHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
        pass

    parser = argparse.ArgumentParser(
        description='Download and run WinML test artifacts',
        epilog='''The default arguments run the latest redist build from the Nuget WindowsAI Pipeline pipeline.

To run tests from the inbox build, use:
    --organization_url https://microsoft.visualstudio.com/ --project WindowsAI --pipeline_id 39810 --artifact_glob arm64-win-inbox-Release''',
        formatter_class=Formatter)
    parser.add_argument('--organization_url', help='URL of the organization', default='https://aiinfra.visualstudio.com')
    parser.add_argument('--project', help='project to fetch artifact from', default='Lotus')
    parser.add_argument(
        '--pat',
        help='Personal Access Token (see '
            'https://docs.microsoft.com/en-us/azure/devops/organizations/accounts/use-personal-access-tokens-to-authenticate'
            'for instructions to create one)')
    parser.add_argument('--artifact_glob', help='glob that matches artifact names', default='arm')
    parser.add_argument('--test_glob', help='glob that matches test executables', default='test_artifact/*_test_*.exe')
    parser.add_argument('test_parameters', help='parameters passed to the test runner', nargs='*')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--build_id', type=int, help='test a specific build')
    group.add_argument('--pipeline_id', type=int,
                       help='test the latest build from given pipeline definition id', default=797)

    args = parser.parse_args()
    if not args.pat:
        args.pat = getpass.getpass('Input your Personal Access Token:')
    return args


def download(url, output_filename, username='', password=''):
    """Download a file to disk using HTTP basic auth"""
    with requests.get(url, auth=HTTPBasicAuth(username, password), stream=True) as request, \
            open(output_filename, 'wb') as f:
        shutil.copyfileobj(request.raw, f)


def get_artifacts(connection, project, build_id=None, pipeline_id=None):
    """Return a dictionary {artifact: download URL}"""
    assert(build_id or pipeline_id)
    client = connection.clients_v6_0.get_build_client()
    if build_id is None:
        build_id = client.get_latest_build(project, pipeline_id, branch_name=None).id
    print(f'Getting artifacts from build {connection.base_url}/{project}/_build/results?buildId={build_id}')
    artifacts = client.get_artifacts(project, build_id)
    return {artifact.name: artifact.resource.download_url for artifact in artifacts}


def main():
    args = parse_args()
    connection = azure.devops.connection.Connection(base_url=args.organization_url,
                                                    creds=msrest.authentication.BasicAuthentication('', args.pat))
    artifacts = get_artifacts(connection, args.project, args.build_id, args.pipeline_id)

    artifact_regex = re.compile(args.artifact_glob)
    platform_artifacts = {name: url for name, url in artifacts.items() if artifact_regex.search(name)}
    if not platform_artifacts:
        raise RuntimeError(f'No artifacts match {args.artifact_glob}')
    for name, url in platform_artifacts.items():
        print(f'Downloading {name}...')
        download(url, f'{name}.zip', password=args.pat)
    for name in platform_artifacts:
        print(f'Extracting {name}...')
        with zipfile.ZipFile(f'{name}.zip', 'r') as zip_archive:
            zip_archive.extractall(name)

        tests = list(Path(name, name).glob(args.test_glob))
        if not tests:
            raise RuntimeError(f'No tests found in {name}')
        print(f'Found tests in {name}: {[str(test) for test in tests]}')
        for test in tests:
            print(f'Executing {test}...')
            interpreter = ()
            if test.suffix == '.ps1':
                interpreter = ('powershell.exe', '-File')
            subprocess.run([*interpreter, test.name, *args.test_parameters], check=True, cwd=test.parent)


if __name__ == '__main__':
    main()
