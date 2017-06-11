//
//  Object.h
//  RayT
//
//  Created by xiewei on 17/3/2.
//  Copyright © 2017年 xiewei. All rights reserved.
//

#ifndef Object_h
#define Object_h

#ifndef M_PI
#define M_PI       3.14159265358979323846  
#endif // !1


#ifdef _WINDOWS
#include "erand.h"
#endif

#include <stdio.h>
#include "glm/glm.hpp"
#include <string>

#define MIN_F 1.0e-3

typedef glm::dvec3 vec3d;
typedef glm::fvec3 vec3f;
typedef vec3d      gvec3;

enum Refl_t { DIFF, SPEC, REFR };  // material types, used in radiance()
const double eps = 1.0e-4;
struct TRay {
    gvec3 ori;
    gvec3 dir;
    TRay(gvec3& o, gvec3& d)
    : ori(o)
    , dir(d)
    {}
};

class Scene;
class CImage;
class Object {
public:
    Object();
    Object(const gvec3& pos, const gvec3& emission, const gvec3& color, Refl_t refl);
    // 求交函数
    virtual bool intersect(const TRay& ray, double& t);
    virtual gvec3 radiance(TRay& ray, double t, int depth, unsigned short* Xi);
    void setScene(Scene* scene);
    void setEntityName(const std::string& name);
    const std::string& getName();
protected:
    CImage* mImage;
    Scene*  mScene;
    std::string mEntityName;
    
    Refl_t mRefl;
    gvec3 mPos;
    gvec3 mEmission;
    gvec3 mColor;
};

#endif /* Object_h */
