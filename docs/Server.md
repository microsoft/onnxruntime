## Build ONNX Runtime Server on Linux
Read more about ONNX Runtime Server [here](./ONNX_Runtime_Server_Usage.md).

### Prerequisites

1. [golang](https://golang.org/doc/install)
2. [grpc](https://github.com/grpc/grpc/blob/master/BUILDING.md). Please be aware that the docs at "[https://grpc.io/docs/quickstart/cpp/](https://grpc.io/docs/quickstart/cpp/)" is outdated, because building with make on UNIX systems is deprecated.
3. [re2](https://github.com/google/re2)
4. cmake
5. gcc and g++
6. onnxruntime C API binaries. Please get it from [github releases](https://github.com/microsoft/onnxruntime/releases) then extract it to your "/usr" or "/usr/local" folder.

See [install_server_deps.sh](../tools/ci_build/github/linux/docker/scripts/install_server_deps.sh) for more details.

### Build Instructions
```
cd server
mkdir build
cmake -DCMAKE_BUILD_TYPE=Debug ..
make
```

ONNX Runtime Server supports sending logs to [rsyslog](https://www.rsyslog.com/) daemon. To enable it, please run the cmake command with an additional parameter: `-Donnxruntime_USE_SYSLOG=1`.

