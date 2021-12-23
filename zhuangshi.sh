export CUDA_HOME=/usr/local/cuda-11.4
export CUDNN_HOME=/usr/local/cuda-11.4
export CUDACXX=$CUDA_HOME/bin/nvcc

export PATH=/usr/local/mpi:$PATH
export LD_LIBRARY_PATH=/usr/local/mpi:$LD_LIBRARY_PATH
export MPI_CXX_INCLUDE_DIRS=/usr/local/mpi:opt/openmpi-4.0.4/include

pip uninstall onnxruntime_training onnxruntime_training_gpu --yes
rm -rf /opt/conda/lib/python3.8/site-packages/onnxruntime
flavor=Debug
flavor=RelWithDebInfo
root=/bert_ort/pengwa/onnxruntime

rm -rf $root/build/Linux/$flavor/dist/*.whl

./build.sh --config $flavor --use_cuda --enable_training  --build_wheel --parallel 8 --use_mpi --skip_tests --enable_training_torch_interop --cuda_version=11.4 --mpi_home=/usr/local/mpi/
pip install ${root}/build/Linux/$flavor/dist/*.whl

cd /tmp/
python -m onnxruntime.training.ortmodule.torch_cpp_extensions.install
cd ${root}
