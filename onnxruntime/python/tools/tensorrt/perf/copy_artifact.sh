#!/bin/bash

while getopts r:a: parameter
do case "${parameter}"
in 
r) RESULT=${OPTARG};;
a) ARTIFACT=${OPTARG};;
esac
done 

mkdir $ARTIFACT && cp -r $RESULT $ARTIFACT
