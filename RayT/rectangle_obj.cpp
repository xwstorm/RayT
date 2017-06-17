//
//  rectangle_obj.c
//  RayT
//
//  Created by xiewei on 17/4/18.
//  Copyright © 2017年 xiewei. All rights reserved.
//

#include "rectangle_obj.h"
#include "scene.h"

CRectangle::CRectangle(const gvec3& pos,
          const gvec3& right,
          const gvec3& up,
          const gvec3& dir,
          const unsigned int width,
          const unsigned int height,
          const gvec3& emission,
          const gvec3& color,
          Refl_t refl)
: Object(pos, emission, color, refl)
, mWidth(width)
, mHeight(height) {
    mMat = glm::mat4(
                     right.x,   right.y,    right.z,    0.0,
                     up.x,      up.y,       up.z,       0.0,
                     dir.x,     dir.y,      dir.z,      0.0,
                     mPos.x,    mPos.y,     mPos.y,     1.0
                     
    );
}

bool CRectangle::intersect(const TRay &ray, double &t) {
//    glm::mat4 matInverse = glm::transpose(mMat);
    glm::mat4 matInverse = glm::inverse(mMat);
    glm::vec4 rayPos(ray.ori.x, ray.ori.y, ray.ori.z, 1.0f);
    glm::vec4 rayPosI = matInverse * rayPos;
    glm::vec4 rayDir(ray.dir.x, ray.dir.y, ray.dir.z, 0.0f);
    glm::vec4 rayDirI = matInverse * rayDir;
    
    if (fabs(rayDirI.z) < MIN_F) {
        return false;
    }
    double k = -( rayPosI.z / rayDirI.z );
    if (k < 0) {
        return false;
    }
    double x = rayPosI.x + k * rayDirI.x;
    double halfWidth = mWidth / 2.0;
    if (x > halfWidth || x < -halfWidth) {
        return false;
    }
    double y = rayPosI.y + k * rayDirI.y;
    double halfHeight = mHeight / 2.0;
    if (y > halfWidth || y < -halfHeight) {
        return false;
    }
    t = k;
    return true;
}

gvec3 CRectangle::radiance(TRay& ray, double t, int depth, unsigned short* Xi) {
    gvec3 hitPos    = ray.ori + ray.dir * t;
    gvec3 radN(mMat[2][0], mMat[2][1], mMat[2][2]);
//    vec3d radN      = glm::normalize(hitPos - mPos);
    gvec3 normal(mMat[2][0], mMat[2][1], mMat[2][2]);
    // 获取交点处的颜色
    gvec3 color = mColor;
    double maxChannel = fmax(fmax(color.x, color.y), color.z);
    if (++depth > 5) {
        if (erand48(Xi) < maxChannel) {
            color = color * 1.0 / maxChannel;
        } else {
            return mEmission;
        }
    }
    
    switch (mRefl) {
        case DIFF: {
            double r1 = 2 * M_PI * erand48(Xi);
            double r2 = erand48(Xi);
            double r2s= glm::sqrt(r2);
            gvec3 w = normal;
            gvec3 u = glm::cross( (fabs(w.x) > 0.1 ? gvec3(0,1,0) : gvec3(1, 0, 0) ), w);
            u = glm::normalize(u);
            gvec3 v = glm::cross(w, u);
            gvec3 newDir = glm::normalize(u*cos(r1)*r2s + v*sin(r1)*r2s + w*sqrt(1-r2));
            ray.ori = hitPos;
            ray.dir = newDir;
            gvec3 result = mEmission + color * mScene->radiance(ray, depth, Xi);
            float len = glm::length(result);
            if (len > 0.01) {
                float len2 = result.length();
                return result;
            }
            return result;
            break;
        }
        case SPEC:{
            gvec3 reflDir = ray.dir - radN * 2.0 * glm::dot(ray.dir, radN);
            ray.ori = hitPos;
            ray.dir = reflDir;
            return mEmission + color * mScene->radiance(ray, depth, Xi);
            break;
        }
        default:
            break;
    }
    return gvec3();
}
