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

Object::Object(const gvec3& pos, const gvec3& emission, const gvec3& color, Refl_t refl)
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

gvec3 Object::radiance(TRay& ray, double t, int depth, unsigned short* Xi) {
    return gvec3();
}

void Object::setEntityName(const std::string& name) {
    mEntityName = name;
}
const std::string& Object::getName() {
    return mEntityName;
}
