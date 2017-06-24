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

#include "glm/glm.hpp"
typedef glm::dvec3 vec3d;
typedef glm::fvec3 vec3f;
typedef vec3d      gvec3;

enum RefType { REF_DIFF, REF_SPEC, REF_REFR };  // material types, used in radiance()

inline double clamp(double x) { return x < 0 ? 0 : x>1 ? 1 : x; }
inline int toInt(double x) { return int(pow(clamp(x), 1 / 2.2) * 255 + .5); }

#endif /* CImage_h */
