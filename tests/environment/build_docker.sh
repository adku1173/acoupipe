#!/bin/sh	

# build docker image from local dockerfile
# and convert it to singularity image

#docker build --target base -f $(pwd)/../../Dockerfile -t acoupipe-test $(pwd)/../../../acoupipe/ && \
sudo singularity build singularity.sif docker-daemon://acoupipe-test:latest
