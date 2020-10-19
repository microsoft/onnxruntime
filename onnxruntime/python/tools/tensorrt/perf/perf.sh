#!/bin/bash

python3 benchmark.py -r validate
python3 benchmark.py -r benchmark -i random -t 100
