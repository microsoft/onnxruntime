// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import <Foundation/Foundation.h>
#import "FakeRCTBlobManager.h"

@implementation FakeRCTBlobManager

- (instancetype)init {
  if (self = [super init]) {
    _blobs = [NSMutableDictionary new];
  }
  return self;
}

- (NSString *)store:(NSData *)data {
  NSString *blobId = [[NSUUID UUID] UUIDString];
  _blobs[blobId] = data;
  return blobId;
}

- (NSData *)resolve:(NSString *)blobId offset:(long)offset size:(long)size {
  NSData *data = _blobs[blobId];
  if (data == nil) {
    return nil;
  }
  return [data subdataWithRange:NSMakeRange(offset, size)];
}

- (NSDictionary *)testCreateData:(NSData *)buffer {
  NSString* blobId = [self store:buffer];
  return @{
    @"blobId": blobId,
    @"offset": @0,
    @"size": @(buffer.length),
  };
}

- (NSString *)testGetData:(NSDictionary *)data {
  NSString *blobId = [data objectForKey:@"blobId"];
  long size = [[data objectForKey:@"size"] longValue];
  long offset = [[data objectForKey:@"offset"] longValue];
  [self resolve:blobId offset:offset size:size];
  return blobId;
}

@end
