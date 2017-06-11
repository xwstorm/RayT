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
    CSphere();
    CSphere(double radius, const gvec3& pos, const gvec3& emission, const gvec3& color, Refl_t refl);
    
    bool intersect(const TRay& ray, double& t) override;
    gvec3 radiance(TRay& ray, double t, int depth, unsigned short* Xi) override;
protected:
    double mRadius;
};
#endif /* CSphere_h */
