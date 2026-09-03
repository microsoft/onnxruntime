// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

/*
 * This file defines RunOptions Config Keys and format of the Config Values.
 *
 * The Naming Convention for a RunOptions Config Key,
 * "[Area][.[SubArea1].[SubArea2]...].[Keyname]"
 * Such as "ep.cuda.use_arena"
 * The Config Key cannot be empty
 * The maximum length of the Config Key is 128
 *
 * The string format of a RunOptions Config Value is defined individually for each Config.
 * The maximum length of the Config Value is 1024
 */

// Key for enabling shrinkages of user listed device memory arenas.
// Expects a list of semi-colon separated key value pairs separated by colon in the following format:
// "device_0:device_id_0;device_1:device_id_1"
// No white-spaces allowed in the provided list string.
// Currently, the only supported devices are : "cpu", "gpu" (case sensitive).
// If "cpu" is included in the list, DisableCpuMemArena() API must not be called (i.e.) arena for cpu should be enabled.
// Example usage: "cpu:0;gpu:0" (or) "gpu:0"
// By default, the value for this key is empty (i.e.) no memory arenas are shrunk
static const char* const kOrtRunOptionsConfigEnableMemoryArenaShrinkage = "memory.enable_memory_arena_shrinkage";

// Set to '1' to not synchronize execution providers with CPU at the end of session run.
// Per default it will be set to '0'
// Taking CUDA EP as an example, it omit triggering cudaStreamSynchronize on the compute stream.
static const char* const kOrtRunOptionsConfigDisableSynchronizeExecutionProviders = "disable_synchronize_execution_providers";

// Set HTP performance mode for QNN HTP backend before session run.
// options for HTP performance mode: "burst", "balanced", "default", "high_performance",
// "high_power_saver", "low_balanced", "extreme_power_saver", "low_power_saver", "power_saver",
// "sustained_high_performance". Default to "default".
static const char* const kOrtRunOptionsConfigQnnPerfMode = "qnn.htp_perf_mode";

// Set HTP performance mode for QNN HTP backend post session run.
static const char* const kOrtRunOptionsConfigQnnPerfModePostRun = "qnn.htp_perf_mode_post_run";

// Set RPC control latency for QNN HTP backend
static const char* const kOrtRunOptionsConfigQnnRpcControlLatency = "qnn.rpc_control_latency";

// Set QNN Lora Config File for apply Lora in QNN context binary
static const char* const kOrtRunOptionsConfigQnnLoraConfig = "qnn.lora_config";

// Set graph annotation id for CUDA EP. Use with enable_cuda_graph=true.
// The value should be an integer. If the value is not set, the default value is 0 and
// ORT session only captures one cuda graph before another capture is requested.
// If the value is set to -1, cuda graph capture/replay is disabled in that run.
// User are not expected to set the value to 0 as it is reserved for internal use.
static const char* const kOrtRunOptionsConfigCudaGraphAnnotation = "gpu_graph_id";

// Declare that the caller has already started a device graph capture on the compute stream
// that the session's execution provider runs on, and that this Run() must therefore execute
// in record-only mode.
//
// In record-only mode ORT issues the run's work to the capturing stream so the caller's graph
// records it, and ORT does not start a capture of its own, does not replay an ORT-managed
// graph, and performs no host-side synchronization. This is what allows a caller to record
// several sequential Run() calls - including calls on different sessions that share one
// caller-owned stream - into a single device graph that later replays with one launch.
//
// Values:
//   "0" (default) - ORT detects a caller-initiated capture on its own. When a capture is
//                   active the run is recorded; otherwise behavior is unchanged.
//   "1"           - require a caller-initiated capture. Run() fails with a diagnostic if the
//                   provider's compute stream is not capturing, which catches the common
//                   mistake of forgetting to bind the session to the capturing stream.
//
// Requirements when recording (CUDA EP):
//   - this feature is unavailable in minimal builds and when CUDA is loaded as a plugin EP;
//   - the session must be created with the capturing stream as "user_compute_stream";
//   - every buffer bound for the run must stay at a stable device address for the lifetime of
//     the caller's graph, and the caller must destroy its captured graphs before the session;
//   - the run must be warmed up beforehand so no device allocation happens while the stream is
//     capturing, including at the largest batch size the pipeline will use: caches that live
//     outside the arena only grow, and growing one mid-capture fails the capture;
//   - this session's graph must satisfy the same capturability rules ORT-managed capture
//     enforces (no control flow nodes, all compute nodes on the capturing provider). Run()
//     rejects the capture otherwise instead of recording a graph that omits work.
//
// Capture mode: start the capture with cudaStreamCaptureModeThreadLocal, or
// cudaStreamCaptureModeRelaxed if the capturing thread must also drive other streams. Do not
// use cudaStreamCaptureModeGlobal unless the process is otherwise quiescent: in Global mode
// CUDA forbids "potentially unsafe" API calls - allocation, stream and event synchronization,
// and similar - on *every* thread for the whole capture window, so unrelated CUDA activity
// elsewhere in the process will fail or invalidate the capture. ThreadLocal restricts that to
// the capturing thread, which still forbids that thread from synchronizing any other stream;
// Relaxed removes the restriction and is the right choice when the capturing thread also
// services other work.
static const char* const kOrtRunOptionsConfigExternalDeviceGraphCapture = "external_device_graph_capture";
