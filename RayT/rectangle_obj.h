//
//  rectangle_obj.h
//  RayT
//
//  Created by xiewei on 17/4/18.
//  Copyright © 2017年 xiewei. All rights reserved.
//

#ifndef rectangle_obj_h
#define rectangle_obj_h

//#include <stdio.h>
#include "object.h"
class CRectangle : public Object {
public:
    CRectangle(const gvec3& pos,
              const gvec3& right,
              const gvec3& up,
              const gvec3& dir,
              const unsigned int width,
              const unsigned int height,
              const gvec3& emission,
              const gvec3& color,
              Refl_t refl);
    bool intersect(const TRay& ray, double& t) override;
    gvec3 radiance(TRay& ray, double t, int depth, unsigned short* Xi) override;
protected:
    gvec3 mRight;
    gvec3 mUp;
    gvec3 mDir;
    double mWidth;
    double mHeight;
    
    glm::mat4 mMat;
};

#endif /* rectangle_obj_h */
