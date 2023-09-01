// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef FakeRCTBlobManager_h
#define FakeRCTBlobManager_h

#import <React/RCTBlobManager.h>

@interface FakeRCTBlobManager : RCTBlobManager

@property (nonatomic, strong) NSMutableDictionary *blobs;

- (NSString *)store:(NSData *)data;

- (NSData *)resolve:(NSString *)blobId offset:(long)offset size:(long)size;

- (NSDictionary *)testCreateData:(NSData *)buffer;

- (NSString *)testGetData:(NSDictionary *)data;

@end

#endif /* FakeRCTBlobManager_h */
