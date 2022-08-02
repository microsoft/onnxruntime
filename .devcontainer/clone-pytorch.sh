#!/bin/sh

# Check if the PyTorch repository exists
if [ ! -d "pytorch" ]
then
    echo "Cloning pytorch repository"
    git clone https://github.com/pytorch/pytorch
fi
