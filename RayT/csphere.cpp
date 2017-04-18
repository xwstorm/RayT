//
//  CSphere.c
//  RayT
//
//  Created by xiewei on 17/3/3.
//  Copyright © 2017年 xiewei. All rights reserved.
//

#include "csphere.h"
#include "scene.h"
CSphere::CSphere(double radius, const vec3d& pos, const vec3d& emission, const vec3d& color, Refl_t refl)
: mRadius(radius)
, mPos(pos)
, mEmission(emission)
, mColor(color)
, mRefl(refl)
{
}

bool CSphere::intersect(const TRay& ray, double& t) {
    vec3d op = mPos - ray.ori;
    double proj = glm::dot(op, ray.dir);
    double delta = proj * proj - glm::dot(op, op) + mRadius * mRadius;
    if (delta < 0.0) {
        return false;
    }
    delta = glm::sqrt(delta);
    t = proj - delta;
    if (t > eps) {
        return true;
    }
    t = proj + delta;
    if (t > eps) {
        return true;
    }
    t = 0.0;
    return false;
}

vec3d CSphere::radiance(TRay& ray, double t, int depth, unsigned short* Xi) {
    vec3d hitPos    = ray.ori + ray.dir * t;
    vec3d radN      = glm::normalize(hitPos - mPos);
    vec3d normal    = glm::dot(radN, ray.dir) < 0 ? radN : -radN;
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
    if (depth == 1 && mEntityName == "9") {
        depth++;
        depth--;
    }
    
    switch (mRefl) {
        case DIFF:
        {
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
            return result;
        }
        case SPEC:
        {
            vec3d reflDir = ray.dir - radN * 2.0 * glm::dot(ray.dir, radN);
            ray.ori = hitPos;
            ray.dir = reflDir;
            return mEmission + color * mScene->radiance(ray, depth, Xi);
        }
        case REFR:
        {
            vec3d reflDir = ray.dir - radN * 2.0 * glm::dot(ray.dir, radN);
            ray.ori = hitPos;
            ray.dir = reflDir;
            bool into = glm::dot(radN, normal) > 0;
            double nc = 1;
            double nt = 1.5;
            double nnt = into ? nc / nt : nt / nc;
            double ddn = glm::dot(ray.dir, normal);
            double cos2t = 1 - nnt*nnt*(1-ddn*ddn);
            if (cos2t < 0) {
                return mEmission + color * mScene->radiance(ray, depth, Xi);
            }
            vec3d tdir = ray.dir * nnt - radN * ( (into?1:-1) * (ddn*nnt + sqrt(cos2t)));
            tdir = glm::normalize(tdir);
            double a = nt - nc;
            double b = nt + nc;
            double R0 = a*a / (b * b);
            double c = 1 - (into? - ddn : glm::dot(tdir, radN));
            double Re=R0+(1-R0)*c*c*c*c*c;
            double Tr=1-Re;
            double P=.25+.5*Re;
            double RP=Re/P;
            double TP=Tr/(1-P);
            
            TRay refRay(hitPos, reflDir);
            TRay rRay(hitPos, tdir);
            if ( depth > 2 ) {
                if (erand48(Xi) < P) {
                    return mEmission + color * mScene->radiance(refRay, depth, Xi) * RP;
                } else {
                    return mEmission + color * mScene->radiance(rRay, depth, Xi) * TP;
                }
            } else {
                return mEmission + color * (
                mScene->radiance(refRay, depth, Xi) * Re +
                mScene->radiance(rRay, depth, Xi) * Tr );
                                            
            }
            
        }
        default:
            break;
    }
    return vec3d();
}
