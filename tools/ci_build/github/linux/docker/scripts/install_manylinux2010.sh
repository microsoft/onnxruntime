#!/bin/bash
set -e
set -x

while getopts p: parameter_Option
do case "${parameter_Option}"
in
p) PYTHON_VER=${OPTARG};;
esac
done

PYTHON_VER=${PYTHON_VER:=3.5}
CPYTHON_VER=cp${PYTHON_VER//./}

# need to install rpmforge in order to get aria2
curl -fsSLo /tmp/rpmforge.rpm http://repository.it4i.cz/mirrors/repoforge/redhat/el6/en/x86_64/rpmforge/RPMS/rpmforge-release-0.5.3-1.el6.rf.x86_64.rpm
yum -y install /tmp/rpmforge.rpm
rm -f /tmp/rpmforge.rpm

yum -y install openblas-devel zlib-devel curl-devel expat-devel aria2 rsync redhat-lsb-core
yum -y clean all

/opt/python/${CPYTHON_VER}-${CPYTHON_VER}m/bin/python -m venv /opt/onnxruntime-python
source /opt/onnxruntime-python/bin/activate
if [ ! -f /opt/onnxruntime-python/bin/python${PYTHON_VER} ]; then
  ln -s python /opt/onnxruntime-python/bin/python${PYTHON_VER}
fi
python -m pip install --upgrade --force-reinstall pip==19.1.1
python -m pip install --upgrade --force-reinstall numpy==1.15.0
python -m pip install --upgrade --force-reinstall requests==2.21.0
python -m pip install --upgrade --force-reinstall wheel==0.31.1
python -m pip install --upgrade --force-reinstall setuptools==41.0.1
python -m pip install --upgrade --force-reinstall pytest==4.6.2

ls -al /opt/onnxruntime-python/bin

echo "#!/bin/sh" > /opt/entrypoint.sh
echo "source /opt/onnxruntime-python/bin/activate" >> /opt/entrypoint.sh
echo "exec \"$@\"" >> /opt/entrypoint.sh

mkdir -p $HOME/.aria2
echo "ca-certificate=/opt/_internal/certs.pem" > $HOME/.aria2/aria2.conf
