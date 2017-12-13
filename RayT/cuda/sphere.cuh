#include "object.cuh"
struct CUSphere : public CUObject
{
    double mRadius;
    CUSphere() {};
    CUSphere(double radius, const gvec3& pos, const gvec3& emission, const gvec3& color, RefType refl)
        : mRadius(radius)
    {
        mPos = pos;
        mEmission = emission;
        mColor = color;
        mRefl = refl;

    }
    __device__ bool intersect(const CURay& ray, double& t) {
        gvec3 op = mPos - ray.ori;
        double proj = glm::dot(op, ray.dir);
        double delta = proj * proj - glm::dot(op, op) + mRadius * mRadius;
        if (delta < 0.0) {
            return false;
        }
        delta = glm::sqrt(delta);
        t = proj - delta;
        if (t > cuEps) {
            return true;
        }
        t = proj + delta;
        if (t > cuEps) {
            return true;
        }
        t = 0.0;
        return false;
    }
};
__device__ gvec3 radiance_device(CUSphere* sphereArr, int size, CURay& ray, int depth, curandState* localState);