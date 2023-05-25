//
//  FakeRCTBlobManager.h
//  OnnxruntimeModuleTest
//
//  Created by Jhen-Jie Hong on 2023/5/25.
//  Copyright © 2023 Facebook. All rights reserved.
//

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
