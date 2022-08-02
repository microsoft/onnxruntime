#!/bin/sh

# Change into /workspaces directory
cd ..

# Check if the PyTorch repository exists
if [ ! -d "pytorch" ]
then
    echo "Cloning pytorch repository"
    git clone https://github.com/pytorch/pytorch
fi
