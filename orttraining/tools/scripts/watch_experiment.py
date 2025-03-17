import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Thread  # noqa: F401

from azureml._run_impl.run_watcher import RunWatcher
from azureml.core import Experiment, Run, Workspace  # noqa: F401
from requests import Session

parser = argparse.ArgumentParser()
parser.add_argument(
    "--subscription", type=str, default="ea482afa-3a32-437c-aa10-7de928a9e793"
)  # AI Platform GPU - MLPerf
parser.add_argument(
    "--resource_group", type=str, default="onnx_training", help="Azure resource group containing the AzureML Workspace"
)
parser.add_argument(
    "--workspace", type=str, default="ort_training_dev", help="AzureML Workspace to run the Experiment in"
)
parser.add_argument("--experiment", type=str, default="BERT-ONNX", help="Name of the AzureML Experiment")
parser.add_argument("--run", type=str, default=None, help="The Experiment run to watch (defaults to the latest run)")

parser.add_argument("--remote_dir", type=str, default=None, help="Specify a remote directory to sync (read) from")
parser.add_argument("--local_dir", type=str, default=None, help="Specify a local directory to sync (write) to")
args = parser.parse_args()

# Validate
if (args.remote_dir and not args.local_dir) or (not args.remote_dir and args.local_dir):
    print("Must specify both remote_dir and local_dir to sync files from Experiment")
    sys.exit()

# Get the AzureML Workspace the Experiment is running in
ws = Workspace.get(name=args.workspace, subscription_id=args.subscription, resource_group=args.resource_group)

# Find the Experiment
experiment = Experiment(workspace=ws, name=args.experiment)

# Find the Run
runs = [r for r in experiment.get_runs()]

if len(runs) == 0:
    print(f"No runs found in Experiment '{args.experiment}'")
    sys.exit()

run = runs[0]
if args.run is not None:
    try:
        run = next(r for r in runs if r.id == args.run)
    except StopIteration:
        print(f"Run id '{args.run}' not found in Experiment '{args.experiment}'")
        sys.exit()

# Optionally start synchronizing files from Run
if args.remote_dir and args.local_dir:
    local_root = os.path.normpath(args.local_dir)
    remote_root = args.remote_dir

    if run.get_status() in ["Completed", "Failed", "Canceled"]:
        print(f"Downloading Experiment files from remote directory: '{remote_root}' to local directory: '{local_root}'")
        files = [f for f in run.get_file_names() if f.startswith(remote_root)]
        for remote_path in files:
            local_path = os.path.join(local_root, os.path.basename(remote_path))
            run.download_file(remote_path, local_path)
    else:
        executor = ThreadPoolExecutor()
        event = Event()
        session = Session()

        print(f"Streaming Experiment files from remote directory: '{remote_root}' to local directory: '{local_root}'")
        watcher = RunWatcher(
            run, local_root=local_root, remote_root=remote_root, executor=executor, event=event, session=session
        )
        executor.submit(watcher.refresh_requeue)

# Block until run completes, to keep updating the files (if streaming)
run.wait_for_completion(show_output=True)
