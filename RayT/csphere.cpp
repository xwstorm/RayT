//
//  CSphere.c
//  RayT
//
//  Created by xiewei on 17/3/3.
//  Copyright © 2017年 xiewei. All rights reserved.
//

#include "csphere.h"

CSphere::CSphere(double radius, glm::vec3& pos, glm::vec3& emission, glm::vec3& color, Refl_t refl)
: mRadius(radius)
, mPos(pos)
, mEmission(emission)
, mColor(color)
, mRefl(refl)
{
}


