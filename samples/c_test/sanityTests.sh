#!/bin/bash

echo 'Compile based EP, relu:'
./build/TestOutTreeEp c relu

echo 'Kernel based EP, relu:'
./build/TestOutTreeEp k relu

echo 'TRT EP, relu:'
./build/TestOutTreeEp t relu

echo 'out tree TRT + In tree cuda, relu:'
./build/TestOutTreeEp tc relu

echo 'out tree TRT + In tree cuda, resnet:'
./build/TestOutTreeEp tc resnet

echo 'out tree TRT + In tree cuda, fast rcnn:'
./build/TestOutTreeEp tc rcnn

echo 'out tree TRT + In tree cuda, tiny yolov3:'
./build/TestOutTreeEp tc tyolo

echo 'out tree TRT + In tree cuda, yolov3:'
./build/TestOutTreeEp tc yolo

echo 'out tree TRT + In tree cuda, control flow:'
./build/TestOutTreeEp tc cf
