#!/usr/bin/env bash
set -e

REPLICAS=${1:-3}

# Build the image and start the cluster using docker compose
# Usage: ./deploy_cluster.sh [replicas]

docker compose up --build --scale aios=$REPLICAS -d
