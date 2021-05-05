// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import <UIKit/UIKit.h>

static void set_test_rootdir(const char* image_path){
    size_t n = strnlen(image_path, 65535);
    for (; n >=0; n--) {
        if (image_path[n] == '/') {
            break;
        }
    }

    char* bundle_dir = (char*)malloc(n + 1);
    if (bundle_dir != NULL) {
        strncpy(bundle_dir, image_path, n);
        bundle_dir[n] = 0;
        chdir(bundle_dir);
        free(bundle_dir);
    }
}

@interface ViewController : UIViewController

@end

@implementation ViewController

- (void)viewDidLoad
{
    [super viewDidLoad];

    self.view.backgroundColor = [UIColor whiteColor];
}

- (void)didReceiveMemoryWarning
{
    [super didReceiveMemoryWarning];
}

@end



@interface AppDelegate : UIResponder <UIApplicationDelegate>

@property (strong, nonatomic) UIWindow *window;

@property (nonatomic, strong) UIViewController *rootViewController;

@end

@implementation AppDelegate


- (BOOL)application:(UIApplication *)application didFinishLaunchingWithOptions:(NSDictionary *)launchOptions
{
    self.window = [[UIWindow alloc] initWithFrame:[UIScreen mainScreen].bounds];
    self.window.tintAdjustmentMode = UIViewTintAdjustmentModeNormal;
    self.window.rootViewController = [[ViewController alloc] init];

    self.window.backgroundColor = [UIColor whiteColor];
    self.window.clipsToBounds = NO;
    [self.window makeKeyAndVisible];

    return YES;
}

- (void)applicationWillResignActive:(UIApplication *)application
{
}

- (void)applicationDidEnterBackground:(UIApplication *)application
{
}

- (void)applicationWillEnterForeground:(UIApplication *)application
{
}

- (void)applicationDidBecomeActive:(UIApplication *)application
{
}

- (void)applicationWillTerminate:(UIApplication *)application
{
}

@end


int main(int argc, char * argv[]) {
    set_test_rootdir(argv[0]);
    int ret = 0;
    @autoreleasepool {
        ret = UIApplicationMain(argc, argv, nil, NSStringFromClass([AppDelegate class]));
    }

    return ret;
}
