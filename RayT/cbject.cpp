//
//  Object.c
//  RayT
//
//  Created by xiewei on 17/3/2.
//  Copyright © 2017年 xiewei. All rights reserved.
//

#include "object.h"
#include "scene.h"
Object::Object()
: mScene(nullptr)
{
}

Object::Object(const vec3d& pos, const vec3d& emission, const vec3d& color, Refl_t refl)
: mPos(pos)
, mEmission(emission)
, mColor(color)
, mRefl(refl) {
}

bool Object::intersect(const TRay& ray, double& t) {
    return false;
}

void Object::setScene(Scene* scene) {
    mScene = scene;
}

vec3d Object::radiance(TRay& ray, double t, int depth, unsigned short* Xi) {
    return vec3d();
}

void Object::setEntityName(const std::string& name) {
    mEntityName = name;
}
const std::string& Object::getName() {
    return mEntityName;
}
