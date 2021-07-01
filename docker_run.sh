#!/bin/bash

if [[ "$1" == "-h" || "$1" == "--help" || "$1" == "" ]]; then
    echo "Usage: $0 <image>"
    exit 2
fi

HERE=$(pwd -P) # Absolute path of current directory
user=`whoami`
uid=`id -u`
gid=`id -g`

IMAGE_NAME="${1}"

DEFAULT_COMMAND="bash"

if [[ $# -gt 0 ]]; then
  shift 1;
  DEFAULT_COMMAND="$@"
  if [[ -z "$1" ]]; then
    DEFAULT_COMMAND="bash"
  fi
fi

DETACHED="-it"

DOCKER_RUN_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

docker_run_params=$(cat <<-END
    -e USER=$user -e UID=$uid -e GID=$gid \
    -v $DOCKER_RUN_DIR:/pointpillars_home \
    -v $HERE:/workspace \
    -w /workspace \
    --rm \
    --network=host \
    ${DETACHED} \
    $IMAGE_NAME \
    $DEFAULT_COMMAND
END
)

nvidia-docker run $docker_run_params
