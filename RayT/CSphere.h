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
    CSphere(double radius, glm::vec3& pos, glm::vec3& emission, glm::vec3& color, Refl_t refl);
    
    glm::vec3 radiance(const Ray& ray, int depth, );
protected:
    double mRadius;
    Refl_t mRefl;
    glm::vec3 mPos;
    glm::vec3 mEmission;
    glm::vec3 mColor;
    
};
#endif /* CSphere_h */
