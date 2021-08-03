#!/bin/bash 

while getopts t: parameter
do case "${parameter}"
in 
t) TRT_CONTAINER=${OPTARG};;
esac
done 

# copying wheel over
id=$(docker create ort-master)
docker cp $id:/code/onnxruntime/build/Linux/Release/dist/ ../
docker rm -v $id

# adding trt container version
wheel_name=$(echo ../dist/*|sed 's/-cp/.'$TRT_CONTAINER'-cp/')
mv ../dist/* $wheel_name
