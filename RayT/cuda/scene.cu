
#include "scene.cuh"
#include "object.cuh"
#include "helper_math.h"
#include <iostream>

int initDeviceScene(int width, int height, CUSphere*& spheres, gvec3*& hostMap, gvec3*& colorMap) {
    const int sphereSize = 8;

    CUSphere* hostSphere = new CUSphere[8]{

        CUSphere(1e5, gvec3(1e5 + 1,40.8,81.6), gvec3(),           gvec3(.75,.25,.25),   REF_DIFF),
        CUSphere(1e5, gvec3(-1e5 + 99,40.8,81.6),gvec3(),           gvec3(.25,.25,.75),   REF_DIFF),
        CUSphere(1e5, gvec3(50,40.8, 1e5),     gvec3(),           gvec3(.75,.75,.75),   REF_DIFF),
        //CUObject(1e5, vec3d(50,40.8,-1e5+170), vec3d(),           vec3d(),              REF_DIFF),
        CUSphere(1e5, gvec3(50, 1e5, 81.6),    gvec3(),           gvec3(.75,.75,.75),   REF_DIFF),
        CUSphere(1e5, gvec3(50,-1e5 + 81.6,81.6),gvec3(),           gvec3(.75,.75,.75),   REF_DIFF),
        CUSphere(16.5,gvec3(27,16.5,47),       gvec3(),           gvec3(1,1,1)*.999,    REF_SPEC),
        CUSphere(16.5,gvec3(73,16.5,78),       gvec3(),           gvec3(1,1,1)*.999,    REF_REFR),
        CUSphere(600, gvec3(50,681.6 - .27,81.6),gvec3(12,12,12),   gvec3(),              REF_DIFF)
    };

    CUSphere* deviceSphere;
    cudaError_t cudaStatus;
    cudaStatus = cudaMalloc((void**)&deviceSphere, sphereSize * sizeof(CUSphere));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(deviceSphere, hostSphere, sphereSize * sizeof(CUSphere), cudaMemcpyHostToDevice);

    cudaStatus = cudaMalloc((void**)&colorMap, width * height * sizeof(gvec3));
    cudaStatus = cudaMemcpy(colorMap, hostMap, width * height * sizeof(gvec3), cudaMemcpyHostToDevice);
    spheres = deviceSphere;
    return sphereSize;
Error:
    //cudaFree(deviceSphere);
    return 0;
}



__global__ void sphereKernel(
    CUSphere* sphereArr,
    int sphereSize,
    CURay cam,
    int depth,
    int sample,
    int width,
    int height,
    gvec3 cx,
    gvec3 cy,
    curandState *const rngStates,
    gvec3* colorMap)
{
    printf("[%d %d %d]\n", blockIdx.x, threadIdx.x, blockDim.x);
    //return;
    int x = threadIdx.x;
    int y = blockIdx.x;
    unsigned int bid = blockIdx.x;
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    gvec3 red(255.0, 0.0, 0.0);
    //red = red * 0.9;
    colorMap[tid] = red * 0.7;
    //return;
    // 每个点在2x2的大小上计算
    gvec3 color;
    int i, sy;
    curandState localState = rngStates[threadIdx.x];
    for (int m = 0; m < GLOBAL_RATE; ++m)
    {
        int y = blockIdx.x * GLOBAL_RATE + m;
        if (y >= height)
        {
            continue;
        }
        for (int n = 0; n < GLOBAL_RATE; ++n)
        {
            int x = threadIdx.x * GLOBAL_RATE + n;
            if (x >= width)
            {
                continue;
            }
            for (sy = 0; sy < 2; sy++)     // 2x2 subpixel rows
            {
                for (int sx = 0; sx < 2; sx++) {        // 2x2 subpixel cols
                    gvec3 tmpRes;
                    gvec3 tmpvec;
                    for (int s = 0; s < sample; ++s)
                    {
                        tmpvec = red * 5.0;
                        //gvec3 tmpvec(5.0, 5.0, 5.0);
                        double r1 = 2 * curand_uniform_double(&localState);
                        double dx = r1 < 1 ? sqrt(r1) - 1 : 1 - sqrt(2 - r1);
                        //double r2 = 2 * curand_uniform_double(&localState);
                        double r2 = r1;
                        double dy = r2 < 1 ? sqrt(r2) - 1 : 1 - sqrt(2 - r2);
                        //continue;
                        // 下面是计算光线的方向
                        // 近裁剪面的x是从-0.5到0.5吗？
                        // 近裁剪面到相机的距离是1
                        //tmpRes.x = 1.0;
                        double value = (((sx + .5 + dx) / 2 + x) / width - 0.5);
                        double cxxx = cx.x;
                        double tx = cx.x * value;
                        gvec3 tmpDir;
                        tmpDir.x = tx;
                        continue;
                        //gvec3 tmpDir = red*value;
                        //gvec3 tmpDir(cx.x*value, cx.y*value, cx.z*value);
                        gvec3 dir = cx*(((sx + 0.5 + dx) / 2.0 + x) / width - 0.5) + cy*(((sy + .5 + dy) / 2.0 + y) / height - .5) + cam.dir;
                        dir = glm::normalize(dir);

                        CURay cuRay(cam.ori, dir);
                        gvec3 ret = radiance_device(sphereArr, sphereSize, cuRay, 0, &localState) * (1.0 / sample);
                        tmpRes += ret;
                    }
                    gvec3 pointValue = gvec3(clamp(tmpRes.x, 0.0f, 1.0f), clamp(tmpRes.y, 0.0f, 1.0f), clamp(tmpRes.z, 0.0f, 1.0f))*.25;//这儿除以4
                    color += pointValue;
                }
            }
            int index = y * width + x;
            colorMap[index] = red;
        }
    }

}



__global__ void sphereKernelTest(
    CUSphere* sphereArr,
    int sphereSize,
    CURay cam,
    int depth,
    int sample,
    int width,
    int height,
    gvec3 cx,
    gvec3 cy,
    curandState *const rngStates,
    gvec3* colorMap)
{
    printf("[%d %d %d]\n", blockIdx.x, threadIdx.x, blockDim.x);
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    gvec3 red(255.0, 0.0, 0.0);
    // 每个点在2x2的大小上计算
    curandState localState = rngStates[tid];
    int index = blockIdx.x * THREAD_DIM + threadIdx.x;
    int yy = (index / OUT_WIDTH) * HEIGHT_STEP;
    int x = index % OUT_WIDTH;

    for (int i = 0; i < HEIGHT_STEP; ++i)
    {
        int y = yy + i;
        gvec3 color;
        for (int sy = 0; sy < 2; sy++)     // 2x2 subpixel rows
        {
            for (int sx = 0; sx < 2; sx++) // 2x2 subpixel cols
            {        
                gvec3 tmpRes;
                for (int s = 0; s < sample; ++s)
                {
                    double r1 = 2 * curand_uniform_double(&localState);
                    double dx = r1 < 1 ? sqrt(r1) - 1 : 1 - sqrt(2 - r1);
                    double r2 = 2 * curand_uniform_double(&localState);
                    double dy = r2 < 1 ? sqrt(r2) - 1 : 1 - sqrt(2 - r2);
                    // 下面是计算光线的方向
                    // 近裁剪面的x是从-0.5到0.5吗？
                    // 近裁剪面到相机的距离是1
                    gvec3 dir = cx*(((sx + 0.5 + dx) / 2.0 + x) / width - 0.5) + cy*(((sy + 0.5 + dy) / 2.0 + y) / height - 0.5) + cam.dir;
                    dir = glm::normalize(dir);
                    CURay cuRay(cam.ori, dir);
                    gvec3 ret = radiance_device(sphereArr, sphereSize, cuRay, 0, &localState) * (1.0 / sample);
                    //gvec3 ret;
                    tmpRes += ret;
                    //tmpRes.x = r1;
                    //tmpRes.y = r2;
                }
                gvec3 pointValue = gvec3(clamp(tmpRes.x, 0.0f, 1.0f), clamp(tmpRes.y, 0.0f, 1.0f), clamp(tmpRes.z, 0.0f, 1.0f))*0.25;//这儿除以4
                color += pointValue;
            }
        }
        int mapIndex = y * OUT_WIDTH + x;
        colorMap[mapIndex] = color;
    }
}



static __global__ void rngSetupStates(
    curandState *rngState,
    int device_id)
{
    // determine global thread id
     int tid = threadIdx.x + blockIdx.x * blockDim.x;
    //curand_init(blockIdx.x + gridDim.x * device_id, threadIdx.x, 0, &rngState[tid]);
    curand_init(blockIdx.x, threadIdx.x, 0, &rngState[tid]);
}

__global__ void testWhy() {
    printf("testWhy [%d %d]\n", blockIdx.x, threadIdx.x);
}

void testKernel() {
    testWhy << <2, 2 >> > ();
    cudaDeviceSynchronize();
}

void startKernel(
    CUSphere* sphereArr,
    int sphereSize,
    CURay& cam,
    int depth,
    int sample,
    int width,
    int height,
    gvec3& cx,
    gvec3& cy,
    curandState *const rngStates,
    int sample_count,
    gvec3* hostMap,
    gvec3* colorMap) {
    //testWhy << <2, 2 >> > ();
    //return;
    cudaError_t cudaStatus = cudaMalloc((void **)&rngStates, height * width * sizeof(curandState));
    if (cudaStatus != cudaSuccess)
    {
        std::cout << "error" << std::endl;
        return;
    }
    int blockDim = (height/ GLOBAL_RATE) + 1;
    int threadDim = (width/ GLOBAL_RATE) + 1;
    printf("begin set up rand state\n");
    rngSetupStates << <BLOCK_DIM, THREAD_DIM >> > (rngStates, 0);
    //rngSetupStates << <height, width >> > (rngStates, 0);
    checkCudaStatus();
    testWhy << <2, 2 >> > ();
    checkCudaStatus();
    cudaDeviceSynchronize();
    checkCudaStatus();
    cudaThreadSetLimit(cudaLimitStackSize, 1024 * 48);

    printf("begin kernel test\n");
    sphereKernelTest << <BLOCK_DIM, THREAD_DIM >> > (sphereArr, sphereSize, cam, 0, sample_count, width, height, cx, cy, rngStates, colorMap);
    checkCudaStatus();
    printf("begin synchronize\n");
    cudaDeviceSynchronize();
    checkCudaStatus();
    cudaError_t copyStatus = cudaMemcpy(hostMap, colorMap, width * height * sizeof(gvec3), cudaMemcpyDeviceToHost);
    if (copyStatus != cudaSuccess)
    {
        printf("cuda error");
    }
}