#!/usr/bin/env bash

set -e

function print_help {
  echo "Build the training Docker image with the given tag and push the image if specified"
  echo "Usage: $(basename "$0") [-ph] [-r IMAGE_REPOSITORY] IMAGE_TAG"
  echo "options:"
  echo "  -h                      Print this help message and exit"
  echo "  -p                      Push the image after built"
  echo "  -t                      Git tag after pushing the image"
  echo "  -g                      Use a GPU image"
  echo "  -r=IMAGE_REPOSITORY     Push the image to a specified repository instead of the default"
}

IMAGE_REPOSITORY="127.0.0.1:5001"
PUSH_IMAGE=false
GIT_TAG=false
USE_GPU_IMAGE=false

while getopts 'hptgr:' OPT; do
  case "$OPT" in
    h)
      print_help
      exit 0
      ;;
    p)
      PUSH_IMAGE=true
      ;;
    t)
      GIT_TAG=true
      ;;
    g)
      USE_GPU_IMAGE=true
      ;;
    r)
      IMAGE_REPOSITORY=$OPTARG;
      ;;
    *)
      print_help;
      exit 1
      ;;
  esac
done

shift $((OPTIND-1));

IMAGE_TAG=$1
if [ -z "$IMAGE_TAG" ]; then
    echo "Missing image tag name"
    print_help
    exit 1
fi

# Training pipeline image
IMAGE_NAME=training
IMAGE_FULL_TAG="$IMAGE_NAME:$IMAGE_TAG"

# Docker file
if [ $USE_GPU_IMAGE = true ]; then
  DOCKERFILE="docker/Dockerfile.train-gpu"
  echo "Using GPU image from: $DOCKERFILE"
else
  DOCKERFILE="docker/Dockerfile.train"
fi

echo "Building $IMAGE_FULL_TAG"
docker build --no-cache -f $DOCKERFILE -t "$IMAGE_FULL_TAG" .

if [ $PUSH_IMAGE = true ]; then
  IMAGE_URL="$IMAGE_REPOSITORY/$IMAGE_NAME:$IMAGE_TAG"
  echo "Pushing image to $IMAGE_URL"
  docker tag "$IMAGE_FULL_TAG" "$IMAGE_URL"
  docker push "$IMAGE_URL"
fi

if [ $GIT_TAG = true ]; then
  git tag "$IMAGE_TAG"
  git push --tags
fi
