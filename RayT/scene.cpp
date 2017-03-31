//
//  scene.c
//  RayT
//
//  Created by xiewei on 17/3/3.
//  Copyright © 2017年 xiewei. All rights reserved.
//

#include "scene.h"
#include "object.h"
Scene::Scene() {
}

Scene::~Scene() {
    for (auto it=mObjs.begin(); it != mObjs.end(); ++it) {
        delete (*it);
    }
}

void Scene::addObject(Object *obj) {
    if (obj != nullptr) {
        mObjs.push_back(obj);
        obj->setScene(this);
    }
}

vec3d Scene::radiance(TRay& ray, int depth, unsigned short *Xi) {
    const float inf = 1e20;
    double minDis = 1e200;
    Object* hitObj = nullptr;
    for (auto it=mObjs.begin(); it != mObjs.end(); ++it) {
        Object* obj = *it;
        double t = 0;
        if (obj->intersect(ray, t)) {
            if (t < minDis) {
                minDis = t;
                hitObj = obj;
            }
        }
    }
    if (minDis<inf && hitObj != nullptr) {
        return hitObj->radiance(ray, minDis, depth++, Xi);
    }
    return vec3d();
}
