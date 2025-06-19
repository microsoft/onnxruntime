---
title: Past present share buffer
description: How to configure the past present share buffer using the ONNX Runtime generate() API
has_children: false
parent: How to
grand_parent: Generate API (Preview)
nav_order: 6
---

# How to configure the past present share buffer

The past present share buffer is an optimization that can be used to save memory and processing time.

When the share buffer is used, the past and present KV cache buffers point to the same memory block.

When the share buffer is not used, the present KV cache buffers are re-allocated before every forward pass of the model, and copied to the past KV cache buffers.

This is represented in the following diagram

![alt text](../../../images/past-present-share-buffer.png)

The size of the KV cache is different depending on the value of the past present share buffer.


## When past_present_share_buffer is true

Past KV caches = Present KV caches = batch_size *  num_key_value_heads + max_length + head_size

Note that the size of the cache is largely determined the value of the max_length parameter.


## When past_present_share_buffer is false

Past KV caches = batch_size * num_key_value_heads * past_sequence_length * head_size

Present KV caches = batch_size *  num_key_value_heads * past_sequence_length + 1 *  head_size
 



