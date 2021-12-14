#!/bin/bash -ex

ort_pkg_path="/home/zhijxu/anaconda3/envs/ortmodule/lib/python3.7/site-packages/onnxruntime"
ort_build_path="/home/zhijxu/local_onnxruntime/onnxruntime/ortmodule_build/Debug"
sudo rm -rf $ort_pkg_path/capi/*.so
sudo ln -s $ort_build_path/libonnxruntime_providers_cuda.so  $ort_pkg_path/capi/libonnxruntime_providers_cuda.so
sudo ln -s $ort_build_path/libonnxruntime_providers_shared.so $ort_pkg_path/capi/libonnxruntime_providers_shared.so
sudo ln -s $ort_build_path/onnxruntime_pybind11_state.so $ort_pkg_path/capi/onnxruntime_pybind11_state.so

cd $ort_build_path
make -j40 onnxruntime_providers_shared onnxruntime_pybind11_state onnxruntime_providers_cuda
