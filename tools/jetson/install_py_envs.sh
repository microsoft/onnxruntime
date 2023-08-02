#!/bin/bash
sudo add-apt-repository -y ppa:deadsnakes/ppa && sudo apt-get update
sudo apt install -y --no-install-recommends build-essential software-properties-common libopenblas-dev python3-pip python3-setuptools python3-wheel
for version in 3.8 3.9 3.10 3.11
do
    sudo apt install -y --no-install-recommends libpython${version}-dev python${version}-dev
	version_no_dot=$(echo $version | tr -d '.')
    python${version} -m venv /home/ort/py${version_no_dot}_venv
    . /home/ort/py${version_no_dot}_venv/bin/activate
    pip install onnx flatbuffers packaging wheel
done
