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
#define GLM_FORCE_CUDA
#include "glm/glm.hpp"
#include <string>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda/cuPrintf.cuh"
#include "curand_kernel.h"
#define MIN_F 1.0e-3


__constant__  const double cuEps = 1.0e-4;
__device__ struct CURay {
    gvec3 ori;
    gvec3 dir;
    __device__ CURay(gvec3 o, gvec3 d)
        : ori(o)
        , dir(d)
    {}
};

int initDevice();
int initObj();

class Scene;
class CImage;
struct CUObject {
        RefType mRefl;
        gvec3 mPos;
        gvec3 mEmission;
        gvec3 mColor;
};
__global__ void initRNG(curandState *const rngStates,
    const unsigned int seed);
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

int checkCudaStatus();
#define GLOBAL_RATE 10 

//#define THREAD_STEP 80      // height
//#define BLOCK_STEP  1280
#define OUT_WIDTH 128
#define OUT_HEIGHT 72

#define HEIGHT_STEP 8
#define THREAD_DIM 9
#define WIDTH_RATE 1//9    // grid dim

#define BLOCK_DIM OUT_WIDTH * WIDTH_RATE