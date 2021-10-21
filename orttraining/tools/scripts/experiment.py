import argparse
import os
import re
import sys

from azure.common.client_factory import get_client_from_cli_profile
from azure.mgmt.containerregistry import ContainerRegistryManagementClient

from azureml.core import Workspace, Experiment, Run, Datastore
from azureml.core.compute import ComputeTarget, AmlCompute

from azureml.core.container_registry import ContainerRegistry
from azureml.train.estimator import Estimator

from azureml.data.azure_storage_datastore import AzureFileDatastore, AzureBlobDatastore
from azureml.core.runconfig import MpiConfiguration, RunConfiguration

parser = argparse.ArgumentParser()
parser.add_argument('--subscription', type=str, default='ea482afa-3a32-437c-aa10-7de928a9e793') # AI Platform GPU - MLPerf
parser.add_argument('--resource_group', type=str, default='onnx_training', help='Azure resource group containing the AzureML Workspace')
parser.add_argument('--workspace', type=str, default='ort_training_dev', help='AzureML Workspace to run the Experiment in')
parser.add_argument('--compute_target', type=str, default='onnx-training', help='AzureML Compute target to run the Experiment on')
parser.add_argument('--experiment', type=str, default='BERT-ONNX', help='Name of the AzureML Experiment')
parser.add_argument('--tags', type=str, default=None, help='Tags to be added to the submitted run (--tag1=value1 --tag2=value2 --tag3=value3)')

parser.add_argument('--datastore', type=str, default='bert_premium', help='AzureML Datastore to be mounted into the Experiment')
parser.add_argument('--train_dir', type=str, default='book/train', help='Path in the AzureML Datastore containing the train files')
parser.add_argument('--test_dir', type=str, default='book/test', help='Path in the AzureML Datastore containing the test files')
parser.add_argument('--train_dir2', type=str, default=None, help='Path in the AzureML Datastore containing the train files for phase 2')
parser.add_argument('--test_dir2', type=str, default=None, help='Path in the AzureML Datastore containing the test files for phase 2')

parser.add_argument('--container', type=str, default='onnxtraining.azurecr.io/azureml/bert:latest-openmpi4.0.0-cuda10.1-cudnn7-ubuntu16.04', help='Docker container to use to run the Experiment')
parser.add_argument('--container_registry_resource_group', type=str, default='onnx_training', help='Azure resource group containing the Azure Container Registry (if not public)')

parser.add_argument('--node_count', type=int, default=1, help='Number of nodes to use for the Experiment. If greater than 1, an MPI distributed job will be run.')
parser.add_argument('--gpu_count', type=int, default=1, help='Number of GPUs to use per node. If greater than 1, an MPI distributed job will be run.')

parser.add_argument('--model_name', type=str, default='bert_L-24_H-1024_A-16_V_30528_optimized_layer_norm', help='Model to be trained (must exist in the AzureML Datastore)')
parser.add_argument('--script_params', type=str, default='', help='Training script parameters (--param1=value1 --param2=value2 --param3=value3)')
args = parser.parse_args()

# Get the AzureML Workspace to run the Experiment in
ws = Workspace.get(name=args.workspace, subscription_id=args.subscription, resource_group=args.resource_group)

# Get the existing AzureML Compute target
compute_target = ComputeTarget(workspace=ws, name=args.compute_target)

# Get the datastore from current workspace
ds = Datastore.get(workspace=ws, datastore_name=args.datastore)

# Construct common script parameters
script_params = {
  '--model_name': ds.path(args.model_name).as_download(),
  '--train_data_dir': ds.path(args.train_dir).as_mount(),
  '--test_data_dir': ds.path(args.test_dir).as_mount(),
}

# Optional phase2 script parameters
if args.train_dir2:
  script_params['--train_data_dir_phase2'] = ds.path(args.train_dir2).as_mount()
if args.test_dir2:
  script_params['--test_data_dir_phase2'] = ds.path(args.test_dir2).as_mount()

# Allow additional custom script parameters
for params in args.script_params.split(' '):
  key, value = params.split('=')
  script_params[key] = value

# Allow custom tags on the run
tags = {}
if args.tags:
  for tag in args.tags.split(' '):
    key, value = tag.split('=')
    tags[key] = value

# Get container registry information (if private)
container_image = args.container
registry_details = None

acr = re.match('^((\w+).azurecr.io)/(.*)', args.container)
if acr:
  # Extract the relevant parts from the container image
  #   e.g. onnxtraining.azurecr.io/azureml/bert:latest
  registry_address = acr.group(1) # onnxtraining.azurecr.io
  registry_name = acr.group(2) # onnxtraining
  container_image = acr.group(3) # azureml/bert:latest

  registry_client = get_client_from_cli_profile(ContainerRegistryManagementClient, subscription_id=args.subscription)
  registry_credentials = registry_client.registries.list_credentials(args.container_registry_resource_group, registry_name)

  registry_details = ContainerRegistry()
  registry_details.address = registry_address
  registry_details.username = registry_credentials.username
  registry_details.password = registry_credentials.passwords[0].value

# MPI configuration if executing a distributed run
mpi = MpiConfiguration()
mpi.process_count_per_node = args.gpu_count

# AzureML Estimator that describes how to run the Experiment
estimator = Estimator(source_directory='./',
                      script_params=script_params,
                      compute_target=compute_target,
                      node_count=args.node_count, 
                      distributed_training=mpi,
                      image_registry_details=registry_details,
                      use_docker=True,
                      custom_docker_image=container_image,
                      entry_script='train.py',
                      inputs=[ds.path('./').as_mount()]
                      )

# Start the AzureML Experiment
experiment = Experiment(workspace=ws, name=args.experiment)
run = experiment.submit(estimator, tags)
print('Experiment running at: {}'.format(run.get_portal_url()))
