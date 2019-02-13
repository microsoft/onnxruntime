FROM ubuntu:16.04

WORKDIR /client

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libboost-filesystem-dev \
    && rm -rf /var/lib/apt/lists/*
