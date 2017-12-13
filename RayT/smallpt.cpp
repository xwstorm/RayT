#include <math.h>   // smallpt, a Path Tracer by Kevin Beason, 2008
#include <stdlib.h> // Make : g++ -O3 -fopenmp smallpt.cpp -o smallpt
#include <stdio.h>  //        Remove "-fopenmp" for g++ version < 4.2
#include "CSphere.h"
#include "scene.cuh"

#include <math.h>

#include <Windows.h>
#include <iostream>

#ifndef M_PI
#define M_PI       3.14159265358979323846
#endif

#define USE_XW



int initScene(CSphere*& spheres) {
    spheres = new CSphere[8] {
    CSphere(1e5, gvec3(1e5 + 1,40.8,81.6), gvec3(),           gvec3(.75,.25,.25),   REF_DIFF),
    CSphere(1e5, gvec3(-1e5 + 99,40.8,81.6),gvec3(),           gvec3(.25,.25,.75),   REF_DIFF),
    CSphere(1e5, gvec3(50,40.8, 1e5),     gvec3(),           gvec3(.75,.75,.75),   REF_DIFF),
    //CSphere(1e5, vec3d(50,40.8,-1e5+170), vec3d(),           vec3d(),              REF_DIFF),
    CSphere(1e5, gvec3(50, 1e5, 81.6),    gvec3(),           gvec3(.75,.75,.75),   REF_DIFF),
    CSphere(1e5, gvec3(50,-1e5 + 81.6,81.6),gvec3(),           gvec3(.75,.75,.75),   REF_DIFF),
    CSphere(16.5,gvec3(27,16.5,47),       gvec3(),           gvec3(1,1,1)*.999,    REF_SPEC),
    CSphere(16.5,gvec3(73,16.5,78),       gvec3(),           gvec3(1,1,1)*.999,    REF_REFR),
    CSphere(600, gvec3(50,681.6 - .27,81.6),gvec3(12,12,12),   gvec3(),              REF_DIFF)
    };
    return 8;
}

void writeBufferToDisk(const char* fileDir, int width, int height, const gvec3* data) {
    std::string filePath(fileDir);
    filePath += "/image.ppm";
    FILE *f = fopen(filePath.c_str(), "w");         // Write image to PPM file.
    fprintf(f, "P3\n%d %d\n%d\n", width, height, 255);
    for (int i = 0; i < width*height; i++) {
        fprintf(f, "%d %d %d ", toInt(data[i].x), toInt(data[i].y), toInt(data[i].z));
    }
    fclose(f);
}

void openConsole() 
{
    AllocConsole();
    freopen("CONIN$", "r", stdin);
    freopen("CONOUT$", "w", stdout);
    freopen("CONOUT$", "w", stderr);
}

void trace_host(int sample_count, const char* fileDir){
    openConsole();
    std::cout << "This is a test info" << std::endl;

    int width=1024, height=768, samps = sample_count/4; // # samples
    CURay cam(gvec3(50,52,295.6), glm::normalize(gvec3(0, 0,-1))); // cam pos, dir -0.042612
    const double rate = 0.5153f; // 0.5153f; // 这个好像是FOV，是y方向上的角度
	gvec3 cx= gvec3(width*rate/height, 0, 0); // 这个cx是干什么的，为什么要乘以0.5135
	gvec3 cy=glm::normalize(glm::cross(cx,cam.dir))*rate;// cy的最大值就是rate
	gvec3 *c=new gvec3[width*height];
	gvec3 r;

    CSphere* spheres = nullptr;
    int sphereSize = initScene(spheres);
    for (int y=0; y<height; y++){                       // Loop over image rows
		xprintf("\rRendering (%d spp) %5.2f%%", samps * 4, 100.*y / (height - 1) );
        for (unsigned short x=0, seed[3]={0,0,static_cast<unsigned short>(y*y*y)}; x<width; x++)   // Loop cols
        {
            // 每个点在2x2的大小上计算
            for (int sy=0, i=(height-y-1)*width+x; sy<2; sy++)     // 2x2 subpixel rows
            {
                for (int sx=0; sx<2; sx++, r=gvec3()){        // 2x2 subpixel cols
                    // samps是输入的参数再除以4
                    gvec3 tmpRes;
                    for (int s=0; s<samps; s++){
                        // dx, dy是什么，为什么要与随机数搞到一块儿？
                        double r1=2*erand48(seed);
                        double dx=r1<1 ? sqrt(r1)-1: 1-sqrt(2-r1);
                        double r2=2*erand48(seed);
                        double dy=r2<1 ? sqrt(r2)-1: 1-sqrt(2-r2);
                        // 下面是计算光线的方向
                        // 近裁剪面的x是从-0.5到0.5吗？
                        // 近裁剪面到相机的距离是1
                        gvec3 dir = cx*( ( (sx+.5 + dx)/2 + x)/width - 0.5) + cy*( ( (sy+.5 + dy)/2 + y)/height - .5) + cam.dir;
                        dir = glm::normalize(dir);

                        {
                            gvec3 ori(cam.ori.x, cam.ori.y, cam.ori.z);
                            gvec3 dir(dir.x, dir.y, dir.z);
                            dir = glm::normalize(dir);
                            
                            TRay cuRay(cam.ori, dir);
                            gvec3 ret = radiance_host(spheres, sphereSize, cuRay, 0, seed) * (1.0 / samps);
                            tmpRes += ret;
                        }
                    }

					gvec3 pointValue = gvec3(clamp(tmpRes.x),clamp(tmpRes.y),clamp(tmpRes.z))*.25;//这儿除以4

                    pointValue = pointValue + c[i];
                    c[i] = pointValue;
                }
            }
        }
    }
    writeBufferToDisk(fileDir, width, height, c);
    return;
}


void trace_device(int sample_count, const char* fileDir) {
    printf("init device\n");
    initDevice();
    //testKernel();
    //return;
    //openConsole();

    int width = OUT_WIDTH, height = OUT_HEIGHT, samps = sample_count / 4; // # samples
    CURay cam(gvec3(50, 52, 295.6), glm::normalize(gvec3(0, 0, -1))); // cam pos, dir -0.042612
    const double rate = 0.5153f; // 0.5153f; // 这个好像是FOV，是y方向上的角度
    gvec3 cx = gvec3(width*rate / height, 0, 0); // 这个cx是干什么的，为什么要乘以0.5135
    gvec3 cy = glm::normalize(glm::cross(cx, cam.dir))*rate;// cy的最大值就是rate
    gvec3 *hostMap = new gvec3[width*height];
    gvec3 r;

    CUSphere* spheres = nullptr;
    gvec3* colorMap = nullptr;
    //int sphereSize = 8;
    printf("begin init scene\n");
    int sphereSize = initDeviceScene(width, height, spheres, hostMap, colorMap);
    printf("begin kernel\n");
    startKernel(spheres, sphereSize, cam, 0, sample_count, width, height, cx, cy, nullptr, sample_count, hostMap, colorMap);

    writeBufferToDisk(fileDir, width, height, hostMap);
    return;
}


void trace(int sampleCount, const char* dir, bool isDevice) {
    if (isDevice)
    {
        trace_device(sampleCount, dir);
    }
    else
    {
        trace_host(sampleCount, dir);
    }
}