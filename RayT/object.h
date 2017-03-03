//
//  Object.h
//  RayT
//
//  Created by xiewei on 17/3/2.
//  Copyright © 2017年 xiewei. All rights reserved.
//

#ifndef Object_h
#define Object_h

#include <stdio.h>
enum Refl_t { DIFF, SPEC, REFR };  // material types, used in radiance()
class CImage;
class Object {
public:
    Object();
    
    // 求交函数
    
protected:
    CImage* mImage;
};

#endif /* Object_h */
