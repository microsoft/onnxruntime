# !/bin/bash

while true;do wget -T 15 -c "$1" && break;done
FILE="$(basename -- "$1")"
unzip $FILE.zip
