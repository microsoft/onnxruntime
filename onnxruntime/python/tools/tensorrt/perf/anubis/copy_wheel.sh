#!/bin/bash 

while getopts t: parameter
do case "${parameter}"
in 
t) TRT_CONTAINER=${OPTARG};;
esac
done 

# copying wheel over
<<<<<<< HEAD
id=$(sudo docker create ort-master)
sudo docker cp $id:/code/onnxruntime/build/Linux/Release/dist/ ../
sudo docker rm -v $id

# adding trt container version
wheel_name=$(echo ../dist/*|sed 's/-cp/.'$TRT_CONTAINER'-cp/')
sudo mv ../dist/* $wheel_name
=======
id=$(docker create ort-master)
docker cp $id:/code/onnxruntime/build/Linux/Release/dist/ ../
docker rm -v $id

# adding trt container version
wheel_name=$(echo ../dist/*|sed 's/-cp/.'$TRT_CONTAINER'-cp/')
mv ../dist/* $wheel_name
>>>>>>> 6ecf626a9c491caffe3bd481d638b27d14534a6f
