// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

/*
 * This file defines SessionOptions Config Keys and format of the Config Values.
 *
 * The Naming Convention for a SessionOptions Config Key,
 * "[Area][.[SubArea1].[SubArea2]...].[Keyname]"
 * Such as "ep.cuda.use_arena"
 * The Config Key cannot be empty
 * The maximum length of the Config Key is 128
 *
 * The string format of a SessionOptions Config Value is defined individually for each Config.
 * The maximum length of the Config Value is 1024
 */

// Influences whether or not the DirectML graph fusion transformer is allowed.
// "0": not disabled (allowed). Graph fusion will be used if the session is not configured to save an optimized model.
// "1": disabled (disallowed). Graph fusion will never be used.
// The default value is "0"
static const char* const kOrtSessionOptionsConfigDisableDmlGraphFusion = "ep.dml.disable_graph_fusion";
static const char* const kOrtSessionOptionsConfigEnableGraphSerialization = "ep.dml.enable_graph_serialization";
