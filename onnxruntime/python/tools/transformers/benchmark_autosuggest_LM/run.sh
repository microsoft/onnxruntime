#!/bin/bash

#####################
#### DO NOT EDIT ####
#####################

if [ -f ~/.bashrc ]; then
    source ~/.bashrc
fi

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
export CURRENT_PATH=$DIR

if [ "$1" == "grpc" ]
then
    python3 $DIR/model/main.py grpc
elif [ "$1" == "http" ]
then
    python3 $DIR/model/main.py http
elif [ "$1" == "udp" ]
then
    python3 $DIR/model/main.py udp
elif [ "$1" == "offline" ]
then
    if [ "$#" == 3 ]
    then
        python3 $DIR/model/main.py offline $2 $3
    fi

    if [ "$#" == 1 ]
    then
        python3 $DIR/model/main.py offline ${_InputFilePath_} ${_OutputFilePath_}
    fi
elif [ "$1" == "offline_profile" ]
then
    if [ "$#" -ne 4 ]
    then
        echo "Usage: ./run.sh offline_profile inputfile outputfile profilename"
    fi

    python3 -m cProfile -o $4.stats $DIR/model/main.py offline $2 $3
    gprof2dot -f pstats $4.stats | dot -Tpng -o $4.png
else
    echo "Usage:"
    echo $1
    echo "1. ./run.sh grpc"
    echo "2. ./run.sh http"
    echo "3. ./run.sh udp"
    echo "3. ./run.sh offline inputfile outputfile"
    echo "4. ./run.sh offline_profile inputfile outputfile profilename"
    exit 1
fi
