#!/bin/bash

# Will run on every container

# clean up files that are generated in earlier runs.
commitid=$1
echo "Download "$commitid
cd /code/
rm binary -rf
rm ort_binary.zip
wget -O ort_binary.zip --no-verbose https://onnxtraining.blob.core.windows.net/philly/binary_${commitid}.tar.gz
tar -xzf ort_binary.zip
mv binary_${commitid} binary
chmod 777 binary -R
rm /tmp/results -rf
cd /code/
echo "Downloaded"
exit 0