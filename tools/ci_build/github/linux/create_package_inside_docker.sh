#!/bin/bash
#This script uses system provided eigen and protobuf, not the ones in Lotus repo.
export PATH=/usr/lib64/ccache:$PATH
set -e
SCRIPT_DIR="$( dirname "${BASH_SOURCE[0]}" )"
TOP_SRC_DIR=$(realpath $SCRIPT_DIR/../../../..)
echo $TOP_SRC_DIR
VERSION_NUMBER=$(cat $TOP_SRC_DIR/VERSION_NUMBER)
rpmdev-setuptree
tmp_build_dir=$(mktemp -d)
export TMP_SOURCE_DIR=$tmp_build_dir/onnxruntime-$VERSION_NUMBER
mkdir -p $TMP_SOURCE_DIR
(cd $SCRIPT_DIR/../../../.. && git archive --format=tar HEAD | (cd $TMP_SOURCE_DIR && tar xf -))
echo "exporting submodules..."
#TODO: support submodule in submodule
(cd $SCRIPT_DIR/../../../.. && git submodule foreach --recursive "
  DEST_DIR=\$TMP_SOURCE_DIR/\$path;
  echo \$DEST_DIR
  mkdir -p \$DEST_DIR
  git archive --format=tar HEAD | tar -C \$DEST_DIR -xf -
")
tar -cf ~/rpmbuild/SOURCES/onnxruntime.tar -C $tmp_build_dir onnxruntime-$VERSION_NUMBER
rm -rf $tmp_build_dir

/usr/bin/cp /data/lotus/package/rpm/onnxruntime.spec ~/rpmbuild/SPECS
rpmbuild -ba ~/rpmbuild/SPECS/onnxruntime.spec

#Install the packages and test it
dnf install -y /root/rpmbuild/RPMS/x86_64/onnxruntime-*$VERSION_NUMBER-*.rpm
if [ -d /data/onnx ]; then
  onnx_test_runner /data/onnx
fi

/usr/bin/cp /root/rpmbuild/RPMS/x86_64/onnxruntime-*$VERSION_NUMBER-*.rpm /data/a


cd /data/a
#convert rpm to tar
for filename in onnxruntime-*.rpm; do
  tmp_dir=$(mktemp -d)
  #older cpio doesn't support -D
  rpm2cpio $filename | (cd $tmp_dir && cpio -idmv);
  dest_filename="${filename//.rpm/.tar.bz2}"
  dest_filename="${dest_filename//\.fc[0-9][0-9]\./.}"
  tar -jcf $dest_filename -C $tmp_dir . ;
  rm -rf tmp_dir
done
