//
//  CImage.h
//  RayT
//
//  Created by xiewei on 17/3/2.
//  Copyright © 2017年 xiewei. All rights reserved.
//
#pragma once
#ifndef __TRYTRACE_CONFIG__
#define __TRYTRACE_CONFIG__
#ifdef _WINDOWS
#include "erand.h"
#define xprintf(format, ...) \
        print_log(format, __VA_ARGS__)
#else
#include "unistd.h"
#define xprintf(format, ...) \
        fprintf(stderr, format, __VA_ARGS__)
#endif


#endif /* CImage_h */
