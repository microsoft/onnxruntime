# print every command
# set -o xtrace

# set path
cd orttraining/tools/amdgpu
DOCKERFILE_PATH=Dockerfile.rocm4.1.pytorch

# get tag
DOCKERFILE_NAME=$(basename $DOCKERFILE_PATH)
DOCKERIMAGE_NAME=$(echo "$DOCKERFILE_NAME" | cut -f 2- -d '.')
echo $DOCKERIMAGE_NAME

# build docker
docker build -f $DOCKERFILE_PATH -t $DOCKERIMAGE_NAME .
