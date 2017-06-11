//
//  CSphere.c
//  RayT
//
//  Created by xiewei on 17/3/3.
//  Copyright © 2017年 xiewei. All rights reserved.
//

#include "csphere.h"
#include "scene.h"
//#include <math.h>
CSphere::CSphere(double radius, const gvec3& pos, const gvec3& emission, const gvec3& color, Refl_t refl)
: Object(pos, emission, color, refl)
, mRadius(radius) {
    
}

bool CSphere::intersect(const TRay& ray, double& t) {
    gvec3 op = mPos - ray.ori;
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

gvec3 CSphere::radiance(TRay& ray, double t, int depth, unsigned short* Xi) {
    gvec3 hitPos    = ray.ori + ray.dir * t;
    gvec3 radN      = glm::normalize(hitPos - mPos);
    gvec3 normal    = glm::dot(radN, ray.dir) < 0 ? radN : -radN;
    // 获取交点处的颜色
    gvec3 color = mColor;
    double maxChannel = fmax(fmax(color.x, color.y), color.z);
    if (++depth > 5) {
//        if (erand48(Xi) < maxChannel) {
//            color = color * 1.0 / maxChannel;
//        } else {
//            return mEmission;
//        }
            return mEmission;
    }
    
    switch (mRefl) {
        case DIFF:
        {
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
            return result;
        }
        case SPEC:
        {
            gvec3 reflDir = ray.dir - radN * 2.0 * glm::dot(ray.dir, radN);
            ray.ori = hitPos;
            ray.dir = reflDir;
            return mEmission + color * mScene->radiance(ray, depth, Xi);
        }
        case REFR:
        {
            gvec3 reflDir = ray.dir - radN * 2.0 * glm::dot(ray.dir, radN);
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
            gvec3 tdir = ray.dir * nnt - radN * ( (into?1:-1) * (ddn*nnt + sqrt(cos2t)));
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
    return gvec3();
}
