//
//  rectangle_obj.h
//  RayT
//
//  Created by xiewei on 17/4/18.
//  Copyright © 2017年 xiewei. All rights reserved.
//

#ifndef rectangle_obj_h
#define rectangle_obj_h

#include <stdio.h>
#include "object.h"
class Rectangle : public Object {
public:
    Rectangle(const vec3d& pos,
              const vec3d& right,
              const vec3d& up,
              const vec3d& dir,
              const unsigned int width,
              const unsigned int height,
              const vec3d& emission,
              const vec3d& color,
              Refl_t refl);
    bool intersect(const TRay& ray, double& t) override;
    vec3d radiance(TRay& ray, double t, int depth, unsigned short* Xi) override;
protected:
    vec3d mRight;
    vec3d mUp;
    vec3d mDir;
    double mWidth;
    double mHeight;
    
    glm::mat4 mMat;
};

#endif /* rectangle_obj_h */
