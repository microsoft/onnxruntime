clear

TMP_DIR=/tmp/onnxruntime

# log dir
ROOT_DIR=$(pwd)
BRANCH_NAME=$(git rev-parse --symbolic-full-name --abbrev-ref HEAD)
DEFAULT_LOG_DIR=$ROOT_DIR/log_${BRANCH_NAME}
LOG_DIR="${1:-$DEFAULT_LOG_DIR}"
rm -rf $LOG_DIR
mkdir -p $LOG_DIR
chmod -R 777 $LOG_DIR

sh scripts/amd/clean.sh 2>&1 | tee $LOG_DIR/clean.log
sh scripts/amd/build.sh 2>&1 | tee $LOG_DIR/build.log
sh scripts/amd/test.sh 2>&1 | tee $LOG_DIR/test.log
