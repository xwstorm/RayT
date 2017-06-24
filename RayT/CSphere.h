//
//  CSphere.h
//  RayT
//
//  Created by xiewei on 17/3/3.
//  Copyright © 2017年 xiewei. All rights reserved.
//

#ifndef CSphere_h
#define CSphere_h

#include <stdio.h>
#include "object.h"
#include "glm/glm.hpp"

class CSphere : public Object
{
public:
    double mRadius;
    CSphere() {};
    CSphere(double radius, const gvec3& pos, const gvec3& emission, const gvec3& color, RefType refl)
        : mRadius(radius)
    {
        mPos = pos;
        mEmission = emission;
        mColor = color;
        mRefl = refl;

    }
    bool intersect(const TRay& ray, double& t) {
        gvec3 op = mPos - ray.ori;
        double proj = glm::dot(op, ray.dir);
        double delta = proj * proj - glm::dot(op, op) + mRadius * mRadius;
        if (delta < 0.0) {
            return false;
        }
        delta = glm::sqrt(delta);
        t = proj - delta;
        if (t > eps) {
            return true;
        }
        t = proj + delta;
        if (t > eps) {
            return true;
        }
        t = 0.0;
        return false;
    }
};
gvec3 radiance_host(CSphere* sphereArr, int size, TRay& ray, int depth, unsigned short* Xi);
#endif /* CSphere_h */
