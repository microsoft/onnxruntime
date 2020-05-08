# Deploying the BERT model using Triton Inference Server

## Solution overview

The [NVIDIA Triton Inference Server](https://github.com/NVIDIA/trtis-inference-server) provides a datacenter and cloud inferencing solution optimized for NVIDIA GPUs. The server provides an inference service via an HTTP or gRPC endpoint, allowing remote clients to request inferencing for any number of GPU or CPU models being managed by the server. 
This folder contains detailed performance analysis as well as scripts to run SQuAD fine-tuning on BERT model using Triton Inference Server. 

## Setup

The first step is to train BERT for question answering. The process is the same as in the main readme. 

1. Download the squad dataset with `cd [bert folder]/data/squad/ && bash ./squad_download.sh`. 

2. Build the Docker container with `bash ./scripts/docker/build.sh`. 

3. [train](https://gitlab-master.nvidia.com/dl/JoC/bert_pyt#training-process) your own checkpoint and fine-tune it, or [download](https://ngc.nvidia.com/catalog/models/nvidia:bert_large_pyt_amp_ckpt_squad_qa1_1/files) the already trained and fine-tuned checkpoint from the [NGC](https://ngc.nvidia.com/catalog/models/nvidia:bert_large_pyt_amp_ckpt_squad_qa1_1/files) model repository. 

The checkpoint should be placed in `[bert folder]/checkpoints/<checkpoint>`. By default, the scripts assume `<checkpoint>` is `bert_qa.pt`, therefore, you might have to rename the trained or downloaded models as necessary. 

Note: The following instructions are run from outside the container and call `docker run` commands as required. \
Unless stated otherwise, all the commands below have to be executed from `[bert folder]`. 

## Quick Start Guide

### Deploying the model

The following command exports the checkpoint to `torchscript`, and deploys the Triton model repository. 

`bash ./triton/export_model.sh` 

The deployed Triton model repository will be in `[bert folder]/results/triton_models`. 

Edit `[bert folder]/triton/export_model.sh` to deploy BERT in ONNX format. 
Change the value of `EXPORT_FORMAT` from `ts-script` to `onnx`. Additionally, change the value of `triton_model_name` from `bertQA-ts` to `bertQA-onnx`, respectively. 
Moreover, you may set `precision` to either `fp32` or `fp16`. 

### Running the Triton server

To launch the Triton server, execute the following command. 

`docker run --rm --gpus device=0 --ipc=host --network=host -p 8000:8000 -p 8001:8001 -p 8002:8002 -v $PWD/results/triton_models:/models nvcr.io/nvidia/tritonserver:20.03-py3 trtserver --model-store=/models --log-verbose=1`

Here `device=0,1,2,3` selects GPUs indexed by ordinals `0,1,2` and `3`, respectively. The server will see only these GPUs. If you write `device=all`, then the server will see all the available GPUs. 

By default, the server expects the model repository to be in `[bert folder]/results/triton_models`. 

### Running the custom Triton client

The custom Triton client is found in `[bert folder]/triton/client.py`. 
It may be used once BERT is deployed and the Triton server is running. To try it, do the following steps. 

1. Start the BERT docker container with the following command: \
`docker run -it --rm --ipc=host --network=host -v $PWD/vocab:/workspace/bert/vocab bert:latest` \
Notice, that for the client, no GPU support is necessary. 

2. Move to the triton folder with the following command: \
`cd /workspace/bert/triton/` 

3. Run the client with the following command: \
`python client.py --do_lower_case --version_2_with_negative --vocab_file=../vocab/vocab --triton-model-name=bertQA-ts-script` 

This will send a request to the already running Triton server, which will process it, and return the result to the client. The response will be printed on the screen. 

You may send your own question-context pair for processing, using the `--question` and `--context` flags of client.py, respectively. 
You may want to use the `--triton-model-name` flag to select the model in onnx format. 

### Evaluating the deployed model on Squad1.1

To deploy and evaluate your model, run the following command. 

`bash ./triton/evaluate.sh` 

By default, this will deploy BERT in torchscript format, and evaluate it on Squad1.1. 

You may change the format of deployment by editing `[bert folder]/triton/evaluate.sh`. 
Change the value of `EXPORT_FORMAT` from `ts-script` to `onnx`. Moreover, you may set `precision` to either `fp32` or `fp16`. 

### Generating performance data

To collect performance data, run the following command. 

`bash ./triton/generate_figures.sh` 

By default, this will deploy BERT in `torchscript` format, launch the server, run the perf client, collect statistics and place them in `[bert folder]/results/triton_models/perf_client`. 

You may change the format of deployment by editing `./triton/generate_figures.sh`. Change the value of `EXPORT_FORMAT` from `ts-script` to `onnx`, respectively. 
Moreover, you may set `precision` to either `fp32` or `fp16`. 

## Advanced

### Other scripts

To launch the Triton server in a detached state, run the following command. 

`bash ./triton/launch_triton_server.sh` 

By default, the Triton server is expecting the model repository in `[bert folder]/results/triton_models`. 

To make the machine wait until the server is initialized, and the model is ready for inference, run the following command. 

`bash ./triton/wait_for_triton_server.sh` 

## Performance

The numbers below are averages, measured on Triton, with [static batching](https://docs.nvidia.com/deeplearning/sdk/tensorrt-inference-server-guide/docs/model_configuration.html#scheduling-and-batching). 

| Format | GPUs | Batch size | Sequence length | Throughput - FP32(sequences/sec) | Throughput - mixed precision(sequences/sec) | Throughput speedup (mixed precision/FP32)  |
|--------|------|------------|-----------------|----------------------------------|---------------------------------------------|--------------------------------------------|
|pytorch      | 1 | 1 | 384 | 30.1 | 28.0  | 0.93x | 
|pytorch      | 1 | 8 | 384 | 36.0 | 116.8 | 3.24x | 
|torchscript  | 1 | 1 | 384 | 32.20 | 38.40 | 1.19x | 
|torchscript  | 1 | 8 | 384 | 40.00 | 134.40 | 3.36x | 
|onnx         | 1 | 1 | 384 | 33.30 | 92.00 | 2.76x | 
|onnx         | 1 | 8 | 384 | 42.60 | 165.30 | 3.88x | 

