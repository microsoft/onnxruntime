// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>

namespace onnxruntime {
namespace cuda {
static constexpr int GIST_PACK1_FACTOR = 8;
template <typename T>
void GistBinarizeEncoderImpl(
    const T* input_data,
    bool* output_data,
    const size_t nums_of_elements);

template <typename T>
void GistBinarizeDecoderImpl(
    const bool* input_data,
    T* output_data,
    const size_t nums_of_elements);

template <typename T>
void GistPack1EncoderImpl(
    const T* input_data,
    uint8_t* output_data,
    const size_t nums_of_elements);

template <typename T>
void GistPack1DecoderImpl(
    const uint8_t* input_data,
    T* output_data,
    const size_t nums_of_elements);

template <typename T>
void GistPack8EncoderImpl(
    const T* input_data,
    uint8_t* output_data,
    const size_t nums_of_elements);

template <typename T>
void GistPack8DecoderImpl(
    const uint8_t* input_data,
    T* output_data,
    const size_t nums_of_elements);

template <typename T>
void GistPack16EncoderImpl(
    const T* input_data,
    half* output_data,
    const size_t nums_of_elements);

template <typename T>
void GistPack16DecoderImpl(
    const half* input_data,
    T* output_data,
    const size_t nums_of_elements);

template <typename T>
void GistPackMsfp15EncoderImpl(
    const T* input_data,
    uint8_t* output_data,
    const size_t pre_axis_size,
    const size_t axis_size,
    const size_t tile_size);

template <typename T>
void GistPackMsfp15DecoderImpl(
    const uint8_t* input_data,
    T* output_data,
    const size_t pre_axis_size,
    const size_t axis_size,
    const size_t tile_size);

}  // namespace cuda
}  // namespace onnxruntime
