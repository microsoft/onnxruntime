#!/bin/bash
# Please run this script using run_mem_test_docker.sh
#

set -x

while getopts p:o:l:s: parameter
do case "${parameter}"
in
p) WORKSPACE=${OPTARG};;
o) ORT_BINARY_PATH=${OPTARG};;
l) BUILD_ORT_LATEST=${OPTARG};;
s) ORT_SOURCE=${OPTARG};;
esac
done

ONNX_MODEL_TAR_URL="https://github.com/onnx/models/raw/main/archive/vision/classification/squeezenet/model/squeezenet1.0-7.tar.gz"
MODEL_TAR_NAME="squeezenet1.0-7.tar.gz"
ONNX_MODEL="squeezenet.onnx"
ASAN_OPTIONS="protect_shadow_gap=0:new_delete_type_mismatch=0:log_path=asan.log"

export LD_LIBRARY_PATH=${ORT_BINARY_PATH}
export LIBRARY_PATH=${ORT_BINARY_PATH}

if [ -z ${BUILD_ORT_LATEST} ]
then
    BUILD_ORT_LATEST="true"
fi

if [ -z ${ORT_SOURCE} ]
then
    ORT_SOURCE="/code/onnxruntime/"
fi

if [ ${BUILD_ORT_LATEST} == "true" ]
then
    cd ${ORT_SOURCE}
    git pull
    ./build.sh --parallel --cuda_home /usr/local/cuda --cudnn_home /usr/lib/x86_64-linux-gnu/ --use_tensorrt --tensorrt_home /usr/lib/x86_64-linux-gnu/ \
               --config Release --build_shared_lib --update --build --cmake_extra_defines ONNXRUNTIME_VERSION=$(cat ./VERSION_NUMBER)
fi

cd ${WORKSPACE}

mkdir build
cd build
cp ../squeezenet_calibration.flatbuffers . 

cmake ..
make -j
wget ${ONNX_MODEL_TAR_URL} -O squeezenet1.0-7.tar.gz
tar -xzf ${MODEL_TAR_NAME} --strip-components=1
mv model.onnx ${ONNX_MODEL}
rm ${MODEL_TAR_NAME}
mkdir result

# Run valgrind
echo $(date +"%Y-%m-%d %H:%M:%S") '[valgrind] Starting memcheck with' ${ONNX_MODEL}
valgrind --leak-check=full --show-leak-kinds=all --log-file=valgrind.log ${ORT_SOURCE}/build/Linux/Release/onnxruntime_perf_test -e tensorrt -r 1 ${ONNX_MODEL}
echo $(date +"%Y-%m-%d %H:%M:%S") '[valgrind] Analyzing valgrind log'

found_leak_summary=false
is_mem_leaked=false

while IFS= read -r line
do
  if echo $line | grep -q 'LEAK SUMMARY:'; then
    found_leak_summary=true
  elif $found_leak_summary && echo $line | grep -q 'definitely lost:'; then
    bytes_lost=$(echo $line | grep -o -E '[0-9,]+ bytes')
    blocks_lost=$(echo $line | grep -o -E '[0-9]+ blocks')
    echo "Bytes lost: $bytes_lost"
    echo "Blocks lost: $blocks_lost"
    if [ "$blocks_lost" != "0 blocks" ]; then
      echo $(date +"%Y-%m-%d %H:%M:%S") '[valgrind] Memory leak happened when testing squeezenet model! Checking if it is ORT-TRT related'
      is_mem_leaked=true
    fi
    found_leak_summary=false
  fi
done < "valgrind.log"

# Export ORT-TRT memleak detail log if available
if [ "$is_mem_leaked" = "true" ]; then
    awk '
    BEGIN {buffer=""; isDefinitelyLost=0; isOrtTrtRelated=0}

    # substitute "==xxxxx==" with ""
    {sub(/==[0-9]+== /, "")}
    
    # Start caching lines when isDefinitelyLost
    /blocks are definitely lost in loss/ {isDefinitelyLost = 1; buffer=""; isOrtTrtRelated=0}
    
    # Cache this line when isDefinitelyLost and line!=""
    # isOrtTrtRelated=1 when "TensorrtExecutionProvider" is found
    isDefinitelyLost && $0 != "" {buffer = buffer "\n" $0; if($0 ~ /TensorrtExecutionProvider/) {isOrtTrtRelated=1}}
    
    # Stop caching and export buffer when isDefinitelyLost, line=="" and isOrtTrtRelated
    isDefinitelyLost && $0 == "" {isDefinitelyLost = 0; if(isOrtTrtRelated==1) {print buffer}}
    ' valgrind.log > ort_trt_memleak_detail.log

    # Check if any ORT-TRT related memleak info has been parsed
    if [ -s ort_trt_memleak_detail.log ]; then
        mv ort_trt_memleak_detail.log result
	    echo $(date +"%Y-%m-%d %H:%M:%S") '[valgrind] ORT-TRT memleak detail log parsed in CI artifact: ort_trt_memleak_detail.log'
        exit 1
    else
	    rm ort_trt_memleak_detail.log
    fi
fi

mv valgrind.log result

# Run AddressSanitizer 
ASAN_OPTIONS=${ASAN_OPTIONS} ./onnx_memtest

if [ -e asan.log* ]
then
    cat asan.log*
    mv asan.log* result
else
    echo $(date +"%Y-%m-%d %H:%M:%S") "[AddressSanitizer] No memory Leak(s) or other memory error(s) detected." > result/asan.log
fi