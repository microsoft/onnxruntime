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

// Key for disable PrePacking,
// If the config value is set to "1" then the prepacking is disabled, otherwise prepacking is enabled (default value)
#define ORT_SESSION_OPTIONS_CONFIG_DISABLEPREPACKING "session.disable_prepacking"

// A value of "1" means allocators registered in the env will be used. "0" means the allocators created in the session
// will be used. Use this to override the usage of env allocators on a per session level.
#define ORT_SESSION_OPTIONS_CONFIG_USE_ENV_ALLOCATORS "session.use_env_allocators"
