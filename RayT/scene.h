//
//  scene.h
//  RayT
//
//  Created by xiewei on 17/3/3.
//  Copyright © 2017年 xiewei. All rights reserved.
//

#ifndef scene_h
#define scene_h

#include <stdio.h>
#include <vector>
#include "object.h"

class Object;
class Scene {
public:
    Scene();
    ~Scene();
    void addObject(Object* obj);
    vec3d radiance(TRay& ray, int depth, unsigned short *Xi);
protected:
    std::vector<Object*> mObjs;
};
#endif /* scene_h */
