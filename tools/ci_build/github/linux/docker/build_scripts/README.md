All the files in this folder were copied from https://github.com/pypa/manylinux with two tiny changes:

```diff
diff -r /data/bt/os/manylinux/docker/build_scripts/install-entrypoint.sh ./install-entrypoint.sh
23a24,25
> yum install -y yum-plugin-versionlock
> yum versionlock cuda* libcudnn*
\ No newline at end of file
diff -r /data/bt/os/manylinux/docker/build_scripts/install-runtime-packages.sh ./install-runtime-packages.sh
87c87,92
<       TOOLCHAIN_DEPS="devtoolset-9-binutils devtoolset-9-gcc devtoolset-9-gcc-c++ devtoolset-9-gcc-gfortran"
---
>       #Added by @snnn
>       if [ ! -d "/usr/local/cuda-10.2" ]; then
>         TOOLCHAIN_DEPS="devtoolset-9-binutils devtoolset-9-gcc devtoolset-9-gcc-c++ devtoolset-9-gcc-gfortran"
>       else
>         TOOLCHAIN_DEPS="devtoolset-8-binutils devtoolset-8-gcc devtoolset-8-gcc-c++ devtoolset-8-gcc-gfortran"
>       fi
```
