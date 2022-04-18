TMP_DIR=/tmp/onnxruntime
cd $TMP_DIR

cd build/RelWithDebInfo

# ../../tools/ci_build/github/pai/pai_test_launcher.sh
./onnxruntime_test_all "--gtest_filter=GraphTransformationTests.ComputationReductionTransformer_GatherND_E2E"
