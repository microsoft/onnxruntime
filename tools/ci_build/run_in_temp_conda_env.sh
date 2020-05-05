#!/bin/bash

ENV_DIR=$(mktemp -d)
PYTHON_VERSION=${PYTHON_VERSION:-3.6}

# enable "conda activate" from bash script
eval "$(conda shell.bash hook)"

conda create --yes --prefix ${ENV_DIR} python=${PYTHON_VERSION} || exit 1

conda activate ${ENV_DIR}

$@
COMMAND_RESULT=$?

conda deactivate

conda env remove --yes --prefix ${ENV_DIR}

# clean up just in case "conda env remove" doesn't
rm -rf "${ENV_DIR}"

exit $COMMAND_RESULT
