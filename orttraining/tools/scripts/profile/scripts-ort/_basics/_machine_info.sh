#!/bin/bash

echo "########## Get GPU Clock Rate/System Load Start ###############"
nvidia-smi -q -i 0 -d CLOCK | sed  "s/^/container id - ${PHILLY_CONTAINER_INDEX} :/"
nvidia-smi | sed  "s/^/container id - ${PHILLY_CONTAINER_INDEX} :/"
uptime | sed  "s/^/container id - ${PHILLY_CONTAINER_INDEX} :/"
echo "########## Get GPU Clock Rate/System Load End ###############"
