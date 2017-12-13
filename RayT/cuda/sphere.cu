

#include "sphere.cuh"

__device__ gvec3 radiance_device(CUSphere* sphereArr, int size, CURay& ray, int depth, curandState* localState) {
    int minDisIndex = -1;
    double minDis = DBL_MAX;
    //printf("size is %d\n", size);
    //return gvec3(0,0,1);
    for (int i = 0; i < size; ++i)
    {
        double dis;
        if (sphereArr[i].intersect(ray, dis) && dis < minDis)
        {
            minDis = dis;
            minDisIndex = i;
        }
    }
    printf("intersect index %d, dis %f\n", minDisIndex, minDis);
    if (minDisIndex < 0 || minDisIndex >= size)
    {
        return gvec3();
    }
    return gvec3(0,0,1);
    CUSphere& sphere = sphereArr[minDisIndex];
    gvec3 hitPos = ray.ori + ray.dir * minDis;
    gvec3 radN = glm::normalize(hitPos - sphere.mPos);
    gvec3 normal = glm::dot(radN, ray.dir) < 0 ? radN : -radN;
    // 获取交点处的颜色
    gvec3 color = sphere.mColor;
    double maxChannel = fmax(fmax(color.x, color.y), color.z);
    if (++depth > 5) {
        //        if (erand48(Xi) < maxChannel) {
        //            color = color * 1.0 / maxChannel;
        //        } else {
        //            return mEmission;
        //        }
        return sphere.mEmission;
    }

    switch (sphere.mRefl) {
    case REF_DIFF:
    {
        double r1 = 2 * M_PI * curand_uniform_double(localState);
        double r2 = curand_uniform_double(localState);
        double r2s = glm::sqrt(r2);
        gvec3 w = normal;
        gvec3 u = glm::cross((fabs(w.x) > 0.1 ? gvec3(0, 1, 0) : gvec3(1, 0, 0)), w);
        u = glm::normalize(u);
        gvec3 v = glm::cross(w, u);
        gvec3 newDir = glm::normalize(u*cos(r1)*r2s + v*sin(r1)*r2s + w*sqrt(1 - r2));
        ray.ori = hitPos;
        ray.dir = newDir;
        gvec3 result = sphere.mEmission + color * radiance_device(sphereArr, size, ray, depth, localState);
        return result;
    }
    case REF_SPEC:
    {
        gvec3 reflDir = ray.dir - radN * 2.0 * glm::dot(ray.dir, radN);
        ray.ori = hitPos;
        ray.dir = reflDir;
        return sphere.mEmission + color * radiance_device(sphereArr, size, ray, depth, localState);
    }
    case REF_REFR:
    {
        gvec3 reflDir = ray.dir - radN * 2.0 * glm::dot(ray.dir, radN);
        ray.ori = hitPos;
        ray.dir = reflDir;
        bool into = glm::dot(radN, normal) > 0;
        double nc = 1;
        double nt = 1.5;
        double nnt = into ? nc / nt : nt / nc;
        double ddn = glm::dot(ray.dir, normal);
        double cos2t = 1 - nnt*nnt*(1 - ddn*ddn);
        if (cos2t < 0) {
            return sphere.mEmission + color * radiance_device(sphereArr, size, ray, depth, localState);
        }
        gvec3 tdir = ray.dir * nnt - radN * ((into ? 1 : -1) * (ddn*nnt + sqrt(cos2t)));
        tdir = glm::normalize(tdir);
        double a = nt - nc;
        double b = nt + nc;
        double R0 = a*a / (b * b);
        double c = 1 - (into ? -ddn : glm::dot(tdir, radN));
        double Re = R0 + (1 - R0)*c*c*c*c*c;
        double Tr = 1 - Re;
        double P = .25 + .5*Re;
        double RP = Re / P;
        double TP = Tr / (1 - P);

        CURay refRay(hitPos, reflDir);
        CURay rRay(hitPos, tdir);
        if (depth > 2) {
            if (curand_uniform_double(localState) < P) {
                return sphere.mEmission + color * radiance_device(sphereArr, size, refRay, depth, localState) * RP;
            }
            else {
                return sphere.mEmission + color * radiance_device(sphereArr, size, ray, depth, localState) * TP;
            }
        }
        else {
            return sphere.mEmission + color * (
                radiance_device(sphereArr, size, refRay, depth, localState) * Re +
                radiance_device(sphereArr, size, ray, depth, localState) * Tr);

        }

    }
    default:
        break;
    }
    return gvec3();
}
