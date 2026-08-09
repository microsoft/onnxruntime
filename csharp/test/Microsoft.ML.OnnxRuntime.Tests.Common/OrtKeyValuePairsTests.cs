// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using Xunit;

namespace Microsoft.ML.OnnxRuntime.Tests;

public class OrtKeyValuePairsTests
{
    private OrtEnv ortEnvInstance = OrtEnv.Instance();


    [Fact]
    public void CRUD()
    {
        using var kvp = new OrtKeyValuePairs();
        kvp.Add("key1", "value1");
        kvp.Add("key2", "value2");
        kvp.Add("key3", "");  // allowed

        Assert.Equal("value1", kvp.Entries["key1"]);
        Assert.Equal("value2", kvp.Entries["key2"]);
        Assert.Equal("", kvp.Entries["key3"]);
        
        kvp.Remove("key1");
        Assert.False(kvp.Entries.ContainsKey("key1"));

        kvp.Remove("invalid_key");  // shouldn't break

        Assert.Equal(2, kvp.Entries.Count);

        // refresh from the C API to make sure everything is in sync
        kvp.Refresh();
        Assert.Equal(2, kvp.Entries.Count);
        Assert.Equal("value2", kvp.Entries["key2"]);
        Assert.Equal("", kvp.Entries["key3"]);
    }
}
