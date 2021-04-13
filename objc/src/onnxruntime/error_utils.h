#import <Foundation/Foundation.h>

@interface ORTErrorUtils : NSObject
+ (void)saveErrorCode:(int)code
          description:(const char*)description_cstr
              toError:(NSError**)error;
@end
