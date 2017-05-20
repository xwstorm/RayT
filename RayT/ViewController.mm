//
//  ViewController.m
//  RayT
//
//  Created by xiewei on 17/2/18.
//  Copyright © 2017年 xiewei. All rights reserved.
//

#import "ViewController.h"

@implementation ViewController
void trace(int sample_count, const char* fileDir);
- (void)viewDidLoad {
    [super viewDidLoad];

    // Do any additional setup after loading the view.
    trace(2, "/Users/xiewei/Desktop");
}


- (void)setRepresentedObject:(id)representedObject {
    [super setRepresentedObject:representedObject];

    // Update the view, if already loaded.
}


@end
