#include "object.cuh"
#include "sphere.cuh"
void testKernel();
int initDeviceScene(int width, int height, CUSphere*& spheres, gvec3*& hostMap, gvec3*& colorMap);
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
    int sample_cout,
    gvec3* hostMap,
    gvec3* colorMap);

