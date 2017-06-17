//
//  Object.h
//  RayT
//
//  Created by xiewei on 17/3/2.
//  Copyright © 2017年 xiewei. All rights reserved.
//

#pragma once

#ifndef M_PI
#define M_PI       3.14159265358979323846  
#endif // !1


#ifdef _WINDOWS
#include "erand.h"
#endif

#include <stdio.h>
#include "config.h"
#include "glm/glm.hpp"
#include <string>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda/cuPrintf.cuh"
#define MIN_F 1.0e-3

typedef glm::dvec3 vec3d;
typedef glm::fvec3 vec3f;
typedef vec3d      gvec3;

enum RefType { REF_DIFF, REF_SPEC, REF_REFR };  // material types, used in radiance()
//const double eps = 1.0e-4;
struct CURay {
    gvec3 ori;
    gvec3 dir;
    CURay(gvec3& o, gvec3& d)
        : ori(o)
        , dir(d)
    {}
};

int outPutDeviceInfo();
int initObj();

class Scene;
class CImage;
struct CUObject {
        RefType mRefl;
        gvec3 mPos;
        gvec3 mEmission;
        gvec3 mColor;
};
//class CUObject {
//public:
//    CUObject();
//    __device__ CUObject(const gvec3& pos, const gvec3& emission, const gvec3& color, RefType refl);
//    __device__ void outPut();
//    // 求交函数
//    __device__ virtual bool intersect(const CURay& ray, double& t);
//    __device__ virtual gvec3 radiance(CURay& ray, double t, int depth, unsigned short* Xi);
//    //__device__ void setScene(Scene* scene);
//    //__device__ void setEntityName(const std::string& name);
//    //__device__ const std::string& getName();
//protected:
//    //CImage* mImage;
//    //Scene*  mScene;
//    //std::string mEntityName;
//
//    RefType mRefl;
//    gvec3 mPos;
//    gvec3 mEmission;
//    gvec3 mColor;
//};
