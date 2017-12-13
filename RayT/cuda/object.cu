//
//  CUObject.c
//  RayT
//
//  Created by xiewei on 17/3/2.
//  Copyright © 2017年 xiewei. All rights reserved.
//

#include "object.cuh"
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

void checkCUDAError(const char *msg) {

    cudaError_t err = cudaGetLastError();

    if (cudaSuccess != err) {

        fprintf(stderr, "Cuda error: %s: %s.\n", msg,

            cudaGetErrorString(err));

        exit(EXIT_FAILURE);

    }

}
int checkCudaStatus() {
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "Cuda error:%d %s.\n", cudaStatus, cudaGetErrorString(cudaStatus));
        return -1;
    }
    return 0;
}

__device__ int deviceTest() {
    return 0;
}
__global__ void why()
{
    //cuPrintf("cuPrintf %d", 0);
    //cuPrintf();
    //cuPrintf("test");
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

int initDevice() {
    std::cout << "outPutDeviceInfo begin" << std::endl;
    int device_count = 0;
    int device = 0;
    checkCudaErrors(cudaGetDeviceCount(&device_count));
    cudaDeviceProp properties;
    for (int i = 0; i < device_count; ++i)
    {
        checkCudaErrors(cudaGetDeviceProperties(&properties, i));
        std::cout << "maxThreadsPerBlock:" << properties.maxThreadsPerBlock << std::endl;
        std::cout << "maxThreadsDim:" << properties.maxThreadsDim[0] << " " << properties.maxThreadsDim[1] << " " << properties.maxThreadsDim[2] << std::endl;
        std::cout << "maxGridSize:" << properties.maxGridSize[0] << " " << properties.maxGridSize[1] << " " << properties.maxGridSize[2] << std::endl;

        if (properties.major > 3 || (properties.major == 3 && properties.minor >= 5))
        {
            device = i;
            xprintf("Running on GPU %d (%s)", i, properties.name);
            //std::cout << "Running on GPU " << i << " (" << properties.name << ")" << std::endl;
            break;
        }
        std::cout << "GPU " << i << " (" << properties.name << ") does not support CUDA Dynamic Parallelism" << std::endl;
    }
    cudaError_t cudaStatus = cudaSetDevice(device);
    return 0;
}

__global__ void initRNG(curandState *const rngStates,
    const unsigned int seed)
{
    // Determine thread ID
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialise the RNG
    curand_init(seed, tid, 0, &rngStates[tid]);
}