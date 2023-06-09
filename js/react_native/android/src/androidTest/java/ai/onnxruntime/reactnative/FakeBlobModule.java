// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

package ai.onnxruntime.reactnative;

import com.facebook.react.bridge.Arguments;
import com.facebook.react.bridge.JavaOnlyMap;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.ReadableMap;
import com.facebook.react.modules.blob.BlobModule;

public class FakeBlobModule extends BlobModule {

  public FakeBlobModule(ReactApplicationContext context) { super(null); }

  @Override
  public String getName() {
    return "BlobModule";
  }

  public JavaOnlyMap testCreateData(byte[] bytes) {
    String blobId = store(bytes);
    JavaOnlyMap data = new JavaOnlyMap();
    data.putString("blobId", blobId);
    data.putInt("offset", 0);
    data.putInt("size", bytes.length);
    return data;
  }

  public byte[] testGetData(ReadableMap data) {
    String blobId = data.getString("blobId");
    int offset = data.getInt("offset");
    int size = data.getInt("size");
    return resolve(blobId, offset, size);
  }
}
