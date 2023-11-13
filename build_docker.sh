#!/bin/sh

#docker build . --file Dockerfile --target base --build-arg TOKEN="xfkM4ZhWP-LSRMWg2JWu" --tag adku1173/acoupipe:dev

#DOCKER_BUILDKIT=1 docker build . --file Dockerfile --target dev --build-arg TOKEN="xfkM4ZhWP-LSRMWg2JWu" --tag adku1173/acoupipe:dev-dev
DOCKER_BUILDKIT=1 docker build . --file Dockerfile --target base --build-arg TOKEN="xfkM4ZhWP-LSRMWg2JWu" --tag adku1173/acoupipe:dev-base
