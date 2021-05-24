#!/bin/bash

orange=`tput setaf 3`
reset_color=`tput sgr0`

export ARCH=`uname -m`

echo "Running on ${orange}${ARCH}${reset_color}"

if [ "$ARCH" == "x86_64" ] 
then
    ARGS="--ipc host --runtime=nvidia -e NVIDIA_DRIVER_CAPABILITIES=all"
elif [ "$ARCH" == "aarch64" ] 
then
    ARGS="--runtime nvidia"
else
    echo "Arch ${ARCH} not supported"
    exit
fi

dir_of_repo=${PWD%/*} 
dir_of_dataset=$1

xhost +local:docker
docker run -it --rm \
	$ARGS \
        --env="DISPLAY=$DISPLAY" \
        --env="QT_X11_NO_MITSHM=1" \
        --privileged \
        --name hierarchical_localization \
        --net "host" \
        -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
	-v $dir_of_repo:/home:rw \
	-v $dir_of_dataset:/datasets:rw \
	--gpus all \
	-p 8888:8888 \
	hloc:latest

