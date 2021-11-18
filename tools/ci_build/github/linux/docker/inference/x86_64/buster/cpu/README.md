
First, on your raspberry pi please install python3-numpy and python3-dev packages. Then check if `/usr/include/numpy` folder exists. If not, check if `/usr/include/python3.7m/numpy` exists. If you find it, copy it to `/usr/include`.

Then copy the root filesystem of your raspberry pi to your build machine(which has a 64-bit x86 CPU). From now on, we assume the copied files are at '/data/piroot/'. 

Then on your build machine, switch to this folder and use the following command to build a docker image for buster x86_64.
```
docker build -t buster .
```

Then, run it

```
mkdir build
docker run -it -v `pwd`/build:/build -v /data/src/onnxruntime:/onnxruntime_src  -v /data/piroot:/data/piroot   --rm buster
```
Inside the docker, run
```
cmake -DCMAKE_BUILD_TYPE=Release -Dprotobuf_WITH_ZLIB=OFF -DCMAKE_TOOLCHAIN_FILE=/onnxruntime_src/cmake/rpi_toolchain.cmake  -Donnxruntime_ENABLE_PYTHON=ON -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPython_EXECUTABLE=/usr/bin/python3 -DPython_ROOT_DIR=/data/piroot/usr -Donnxruntime_BUILD_SHARED_LIB=OFF -Donnxruntime_DEV_MODE=OFF -DONNX_CUSTOM_PROTOC_EXECUTABLE=/usr/bin/protoc -DPython_NumPy_INCLUDE_DIR=/data/piroot/usr/include -DPython_INCLUDE_DIR=/data/piroot/usr/include/python3.7m /onnxruntime_src/cmake
```

When the build is finished, copy setup.py from onnxruntime's source dir to the build dir, then copy the build dir to your raspberry pi device. Or if they are in the same local network, you can use nfs or samba to share the files then you don't need to do the copy.

On the device, switch to the build dir and run:
```
python3 setup.py bdist_wheel 
```
