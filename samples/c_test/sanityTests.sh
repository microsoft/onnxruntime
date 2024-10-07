#!/bin/bash

echo 'Compile based EP, relu:'
./TestOutTreeEp c relu

echo 'Kernel based EP, relu:'
./TestOutTreeEp k relu

echo 'TRT EP, relu:'
./TestOutTreeEp t relu

echo 'out tree TRT + In tree cuda, relu:'
./TestOutTreeEp tc relu

echo 'out tree TRT + In tree cuda, resnet:'
./TestOutTreeEp tc resnet

echo 'out tree TRT + In tree cuda, fast rcnn:'
./TestOutTreeEp tc rcnn

echo 'out tree TRT + In tree cuda, tiny yolov3:'
./TestOutTreeEp tc tyolo

echo 'out tree TRT + In tree cuda, yolov3:'
./TestOutTreeEp tc yolo

echo 'out tree TRT + In tree cuda, control flow:'
./TestOutTreeEp tc cf
