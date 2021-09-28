#!/bin/bash 

while getopts t:i: parameter
do case "${parameter}"
in 
t) TRT_CONTAINER=${OPTARG};;
i) ANUBIS_IMAGE=${OPTARG};;
esac
done 

# copying wheel over
id=$(docker create $ANUBIS_IMAGE)
docker cp $id:/code/onnxruntime/build/Linux/Release/dist/ ../
docker rm -v $id

# adding trt container version
wheel_name=$(echo ../dist/*|sed 's/-cp/.'$TRT_CONTAINER'-cp/')
mv ../dist/* $wheel_name
