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
class CSphere : public Object {
public:
    static int hitCount;
    CSphere();
    CSphere(double radius, const vec3d& pos, const vec3d& emission, const vec3d& color, Refl_t refl);
    
    bool intersect(const TRay& ray, double& t) override;
    vec3d radiance(TRay& ray, double t, int depth, unsigned short* Xi) override;
protected:
    double mRadius;
    Refl_t mRefl;
    vec3d mPos;
    vec3d mEmission;
    vec3d mColor;
};
#endif /* CSphere_h */
