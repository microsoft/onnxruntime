# --------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------
# Dockerfile for ONNX Runtime Android package build environment

FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

# install utilities and ORT dependencies
RUN apt-get update && apt-get install --yes --no-install-recommends \
  aria2 curl \
  build-essential \
  ca-certificates \
  git \
  ninja-build \
  openjdk-11-jdk-headless \
  python3-dev \
  python3-numpy \
  python3-pip \
  python3-setuptools \
  python3-wheel \
  unzip lsb-release

# cmake
RUN CMAKE_VERSION=3.27.3 && \
  aria2c -q -d /tmp -o cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz \
  --checksum=sha-256=62e7819fe0867658b6ea765a711686d637dce76cdf6eb0a6b0f1b879e0344fa7 \
  https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz && \
  tar -zxf /tmp/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz --strip=1 -C /usr

# gradle
# Note: A system-installed gradle is not required in more recent versions of ORT which use an included gradle wrapper,
#       but keep it around to support building older versions.
RUN GRADLE_VERSION=6.8.3 && \
  aria2c -q -d /tmp -o gradle-${GRADLE_VERSION}-bin.zip \
  --checksum=sha-256=7faa7198769f872826c8ef4f1450f839ec27f0b4d5d1e51bade63667cbccd205 \
  https://services.gradle.org/distributions/gradle-${GRADLE_VERSION}-bin.zip && \
  mkdir /opt/gradle && \
  unzip -d /opt/gradle /tmp/gradle-${GRADLE_VERSION}-bin.zip && \
  ln -s /opt/gradle/gradle-${GRADLE_VERSION}/bin/gradle /usr/bin

# flatbuffers
RUN python3 -m pip install flatbuffers==2.0

# configure user and set up workspace directory
ARG BUILD_UID=1001
ARG BUILD_USER=onnxruntimedev
RUN adduser --gecos "onnxruntime Build User" --disabled-password --uid ${BUILD_UID} ${BUILD_USER} && \
  mkdir /workspace && \
  chown ${BUILD_USER} /workspace
USER ${BUILD_USER}
WORKDIR /workspace

# install Android SDK and tools
ENV ANDROID_HOME=~/android-sdk
ENV NDK_VERSION=26.0.10792818
ENV ANDROID_NDK_HOME=${ANDROID_HOME}/ndk/${NDK_VERSION}

RUN aria2c -q -d /tmp -o cmdline-tools.zip \
  --checksum=sha-256=0bebf59339eaa534f4217f8aa0972d14dc49e7207be225511073c661ae01da0a \
  https://dl.google.com/android/repository/commandlinetools-linux-9123335_latest.zip && \
  unzip /tmp/cmdline-tools.zip -d /tmp/cmdline-tools && \
  mkdir -p ${ANDROID_HOME}/cmdline-tools && \
  mv /tmp/cmdline-tools/cmdline-tools ${ANDROID_HOME}/cmdline-tools/latest

RUN yes | ${ANDROID_HOME}/cmdline-tools/latest/bin/sdkmanager --licenses
RUN ${ANDROID_HOME}/cmdline-tools/latest/bin/sdkmanager --install \
  "platforms;android-32" \
  "ndk;${NDK_VERSION}"

# get ORT repo
ARG ONNXRUNTIME_REPO=https://github.com/microsoft/onnxruntime.git
ARG ONNXRUNTIME_BRANCH_OR_TAG=main
RUN git clone --single-branch --branch=${ONNXRUNTIME_BRANCH_OR_TAG} --recurse-submodules ${ONNXRUNTIME_REPO} \
  /workspace/onnxruntime

# add scripts
COPY --chown=${BUILD_UID} scripts /workspace/scripts
