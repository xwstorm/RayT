//
//  rectangle_obj.c
//  RayT
//
//  Created by xiewei on 17/4/18.
//  Copyright © 2017年 xiewei. All rights reserved.
//

#include "rectangle_obj.h"
#include "scene.h"

Rectangle::Rectangle(const vec3d& pos,
          const vec3d& right,
          const vec3d& up,
          const vec3d& dir,
          const unsigned int width,
          const unsigned int height,
          const vec3d& emission,
          const vec3d& color,
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

bool Rectangle::intersect(const TRay &ray, double &t) {
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

vec3d Rectangle::radiance(TRay& ray, double t, int depth, unsigned short* Xi) {
    vec3d hitPos    = ray.ori + ray.dir * t;
    vec3d radN(mMat[2][0], mMat[2][1], mMat[2][2]);
//    vec3d radN      = glm::normalize(hitPos - mPos);
    vec3d normal(mMat[2][0], mMat[2][1], mMat[2][2]);
    // 获取交点处的颜色
    vec3d color = mColor;
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
            vec3d w = normal;
            vec3d u = glm::cross( (fabs(w.x) > 0.1 ? vec3d(0,1,0) : vec3d(1, 0, 0) ), w);
            u = glm::normalize(u);
            vec3d v = glm::cross(w, u);
            vec3d newDir = glm::normalize(u*cos(r1)*r2s + v*sin(r1)*r2s + w*sqrt(1-r2));
            ray.ori = hitPos;
            ray.dir = newDir;
            vec3d result = mEmission + color * mScene->radiance(ray, depth, Xi);
            float len = glm::length(result);
            if (len > 0.01) {
                float len2 = result.length();
                return result;
            }
            return result;
            break;
        }
        case SPEC:{
            vec3d reflDir = ray.dir - radN * 2.0 * glm::dot(ray.dir, radN);
            ray.ori = hitPos;
            ray.dir = reflDir;
            return mEmission + color * mScene->radiance(ray, depth, Xi);
            break;
        }
        default:
            break;
    }
    return vec3d();
}
