#!/bin/sh

# Check if the PyTorch repository exists
if [ ! -d "pytorch" ]
then
    echo "Directory does not exist"
    # git clone https://github.com/pytorch/pytorch
fi
