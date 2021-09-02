#!/bin/bash 

apt-get update 
apt-get install -y sudo git bash unattended-upgrades wget
unattended-upgrade

# Install python3
apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    python3-wheel &&\
    cd /usr/local/bin &&\
    ln -s /usr/bin/python3 python &&\
    ln -s /usr/bin/pip3 pip;

pip install --upgrade pip 
pip install setuptools>=41.0.0

