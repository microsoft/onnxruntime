All the files in this folder were copied from https://github.com/pypa/manylinux (commit id 92f447b951e121d4df46fbbc68982a76a0e7dc1b) with two tiny changes:

```diff
diff -r /data/bt/os/manylinux/docker/build_scripts/install-entrypoint.sh ./install-entrypoint.sh
23a24,25
> yum install -y yum-plugin-versionlock
> yum versionlock cuda* libcudnn*
\ No newline at end of file
diff -r /data/bt/os/manylinux/docker/build_scripts/install-runtime-packages.sh ./install-runtime-packages.sh
87c87,95
<       TOOLCHAIN_DEPS="devtoolset-10-binutils devtoolset-10-gcc devtoolset-10-gcc-c++ devtoolset-10-gcc-gfortran"
---
> 
>       #Added by @snnn
>       if [ -d "/usr/local/cuda-10.2" ]; then
>         TOOLCHAIN_DEPS="devtoolset-8-binutils devtoolset-8-gcc devtoolset-8-gcc-c++ devtoolset-8-gcc-gfortran"
>       elif [ -d "/usr/local/cuda-11.1" ]; then
>         TOOLCHAIN_DEPS="devtoolset-9-binutils devtoolset-9-gcc devtoolset-9-gcc-c++ devtoolset-9-gcc-gfortran"
>       else
>         TOOLCHAIN_DEPS="devtoolset-10-binutils devtoolset-10-gcc devtoolset-10-gcc-c++ devtoolset-10-gcc-gfortran"
>       fi
92c100,102
<               yum -y install https://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm
---
>               if ! rpm -q --quiet epel-release ; then
>                 yum -y install https://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm
>               fi
```
