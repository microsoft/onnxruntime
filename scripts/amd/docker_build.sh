# print every command
set -o xtrace

# set path
cd dockerfiles
DOCKERFILE_PATH=Dockerfile.rocm

# get tag
DOCKERFILE_NAME=$(basename $DOCKERFILE_PATH)
DOCKERIMAGE_NAME=$(echo "$DOCKERFILE_NAME" | cut -f 2- -d '.')
echo $DOCKERIMAGE_NAME

# build docker
docker build -f $DOCKERFILE_PATH -t ort_rocm .
