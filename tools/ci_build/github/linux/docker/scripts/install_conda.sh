#!/bin/bash
INSTALL_SCRIPT=/tmp/install_conda.sh
INSTALL_PREFIX=/opt/miniconda3
CONDA_MD5SUM=cbda751e713b5a95f187ae70b509403f
curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-py38_4.8.2-Linux-x86_64.sh -o ${INSTALL_SCRIPT} && \
    echo "${CONDA_MD5SUM}  ${INSTALL_SCRIPT}" | md5sum -c - &&
    chmod 755 ${INSTALL_SCRIPT} && \
    ${INSTALL_SCRIPT} -b -p ${INSTALL_PREFIX} && \
    rm ${INSTALL_SCRIPT}
