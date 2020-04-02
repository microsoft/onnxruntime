# Introduction

ONNX Runtime Trainer is a test feature introduced in the ONNX Runtime engine. This trainer can be used to accelerate the computation of the ops used to train transformer class models.

The ONNX Runtime trainer can be used with your existing Pytorch training code to accelerate execution on NVIDIA GPU clusters.

## Build

Build the ONNX Runtime Training engine to use with NVIDIA GPUs for accelerating the computations.

### Dependencies

This default NVIDIA GPU build requires CUDA runtime libraries installed on the system:

    * The GPU-accelerated CUDA libraries [CUDA 10.1](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1604&target_type=debnetwork)
    * The GPU-accelerated library of primitives for deep neural networks [cuDNN 7.6.2](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#install-linux)
    * The NVIDIA Collective Communications Library (NCCL) multi-GPU and multi-node communication primitives library [NCCL2](https://docs.nvidia.com/deeplearning/sdk/nccl-install-guide/index.html)
    * OpenMPI 4.0.0.0

```
wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.0.tar.gz
tar zxf openmpi-4.0.0.tar.gz
cd openmpi-4.0.0
./configure --enable-orterun-prefix-by-default
make -j $(nproc) all
sudo make install
sudo ldconfig
```

### Get the code and setup the environment

* Checkout this code repo with `git clone https://github.com/microsoft/onnxruntime`

* Environment variables

```
export CUDA_HOME=/usr/local/cuda
export CUDNN_HOME=/usr/local/cuda
export CUDACXX=/usr/local/cuda/bin/nvcc
export LD_LIBRARY_PATH=/bert_ort/openmpipython/lib:$LD_LIBRARY_PATH
export PATH=/bert_ort/openmpi/bin/:$PATH
export LD_LIBRARY_PATH=/bert_ort/openmpi/lib/:$LD_LIBRARY_PATH
export MPI_CXX_INCLUDE_PATH=/bert_ort/openmpi/include/
source /data/intel/impi/2018.3.222/intel64/bin/mpivars.sh
```

### Create the ONNX Runtime wheel

Run `./build.sh --enable_training --use_cuda --config=RelWithDebInfo --build_wheel`

This will produce the `.whl` file in `onnxruntime/build/Linux/RelWithDebInfo/dist` for ONNX Runtime Trainer.

## Use with Pytorch training

You can use the ONNX Runtime Training wheel as the *trainer* in your Pytorch pre-training script. Here is a high-level code fragment to include in your pre-training code:

```
import torch
import onnxruntime.training.pytorch as ort

# Model definition
class Net(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        ...
    def forward(self, x): 
        ...

model = Net(D_in, H, H_out)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
trainer = ort.trainer(model, criterion, optimizer, ...)

# Training Loop
for t in range(1000):
    # forward + backward + weight update 
    loss, y_pred = trainer.step(x, y)
    ...
    
```

> a sample for end-to-end training coming soon
