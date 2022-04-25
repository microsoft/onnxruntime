# clear

# export AMD_OCL_WAIT_COMMAND=1
# export AMD_LOG_LEVEL=3
# export HIP_LAUNCH_BLOCKING=1

# log dir
ROOT_DIR=$(pwd)
BRANCH_NAME=$(git rev-parse --symbolic-full-name --abbrev-ref HEAD)
DEFAULT_LOG_DIR=$ROOT_DIR/log_${BRANCH_NAME}
LOG_DIR="${1:-$DEFAULT_LOG_DIR}"
rm -rf $LOG_DIR
mkdir -p $LOG_DIR
chmod -R 777 $LOG_DIR

TMP_DIR=/tmp/onnxruntime
cd $TMP_DIR

cd build/RelWithDebInfo

# ../../tools/ci_build/github/pai/pai_test_launcher.sh 2>&1 | tee $LOG_DIR/ort_full_test.log

# Failing Tests
# ./onnxruntime_test_all "--gtest_filter=GraphTransformationTests.ComputationReductionTransformer_GatherND_E2E" 2>&1 | tee $LOG_DIR/ort_filter_test.log
# ./onnxruntime_test_all "--gtest_filter=LayerNormTest.BERTLayerNorm" 2>&1 | tee $LOG_DIR/ort_filter_test.log
# ./onnxruntime_test_all "--gtest_filter=GradientCheckerTest.ReduceLogSumExpGrad" # segfault 2>&1 | tee $LOG_DIR/ort_filter_test.log
./onnxruntime_test_all "--gtest_filter=CudaKernelTest.SoftmaxCrossEntropy_SmallSizeTensor" 2>&1 | tee $LOG_DIR/ort_filter_test.log
# ./onnxruntime_test_all "--gtest_filter="  2>&1 | tee $LOG_DIR/ort_filter_test.log


echo "KERNELS:"
cat $LOG_DIR/ort_filter_test.log | grep _ZN
