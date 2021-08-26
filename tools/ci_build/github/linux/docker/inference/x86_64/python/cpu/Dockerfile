# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# This file is used by Python packaging pipeline
FROM quay.io/pypa/manylinux2014_x86_64:latest

ADD scripts /tmp/scripts
RUN cd /tmp/scripts && ./run.sh && cd / && rm -rf /tmp/scripts
