//
//  CUObject.c
//  RayT
//
//  Created by xiewei on 17/3/2.
//  Copyright © 2017年 xiewei. All rights reserved.
//

#include "object.cuh"
#include "scene.h"
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <helper_cuda.h>
#include <helper_string.h>

//
//CUObject::CUObject()
//    //: mScene(nullptr)
//{
//}
//
//__device__ CUObject::CUObject(const gvec3& pos, const gvec3& emission, const gvec3& color, RefType refl)
//    : mPos(pos)
//    , mEmission(emission)
//    , mColor(color)
//    , mRefl(refl) {
//}
//
//__device__ bool CUObject::intersect(const CURay& ray, double& t) {
//    return false;
//}
//
//__device__ void CUObject::outPut() {
//    printf("CUObject outPut");
//}
////
////void CUObject::setScene(Scene* scene) {
////    mScene = scene;
////}
////
//gvec3 CUObject::radiance(CURay& ray, double t, int depth, unsigned short* Xi) {
//    return gvec3();
//}
//
//void CUObject::setEntityName(const std::string& name) {
//    mEntityName = name;
//}
//const std::string& CUObject::getName() {
//    return mEntityName;
//}

//__device__ CUObject* createObj() {
//    //CUObject* obj = new CUObject();
//    //return obj;
//    return nullptr;
//}

__global__ void classBegin() {
    printf("block %d, thread %d", blockIdx.x, threadIdx.x);
    return;
}

__device__ int deviceTest() {
    return 0;
}
__global__ void why()
{
    //cuPrintf("cuPrintf %d", 0);
    cuPrintf();
    cuPrintf("test");
    //cuPrintf("cuPrintf %d %d", 12, 13);
    deviceTest();
    printf("why BLOCK %d launched by the host\n\n\n\n", threadIdx.x);
}

int initObj() 
{
    const int width = 1024;
    const int height = 800;
    const int size = width * height;
    cudaError_t cudaStatus;
    void* retBuffer;
    cudaStatus = cudaMalloc((void**)&retBuffer, size * sizeof(gvec3));
    if (cudaStatus != cudaSuccess)
    {
        xprintf("malloc ret buffer error");
    }
    //CUObject* obj = new CUObject();
    //cudaStatus = cudaMalloc((void**))
    return 0;
}

int outPutDeviceInfo() {
    cudaPrintfInit(1024);
    std::cout << "outPutDeviceInfo begin" << std::endl;
    int device_count;
    int device;
    checkCudaErrors(cudaGetDeviceCount(&device_count));
    for (int i = 0; i < device_count; ++i)
    {
        cudaDeviceProp properties;
        checkCudaErrors(cudaGetDeviceProperties(&properties, i));
        if (properties.major > 3 || (properties.major == 3 && properties.minor >= 5))
        {
            device = i;
            xprintf("Running on GPU %d (%s)", i, properties.name);
            //std::cout << "Running on GPU " << i << " (" << properties.name << ")" << std::endl;
            break;
        }
        std::cout << "GPU " << i << " (" << properties.name << ") does not support CUDA Dynamic Parallelism" << std::endl;
    }
    printf("why");
    why<<<1,2>>>();
    cudaPrintfDisplay(stdout, true);
    classBegin <<<1, 2 >>>();
    return 0;
}