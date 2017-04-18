#include <math.h>   // smallpt, a Path Tracer by Kevin Beason, 2008
#include <stdlib.h> // Make : g++ -O3 -fopenmp smallpt.cpp -o smallpt
#include <stdio.h>  //        Remove "-fopenmp" for g++ version < 4.2
#include "object.h"
#include "scene.h"
#include "csphere.h"
#include <math.h>
#ifndef _WINDOWS
#include "unistd.h"
#endif
#ifndef M_PI
#define M_PI       3.14159265358979323846
#endif

double getRand(unsigned short * in) {
#ifdef _WINDOWS
	return 
#else
	return erand48(in);
#endif
}
#define USE_XW
struct Vec {        // Usage: time ./smallpt 5000 && xv image.ppm
    double x, y, z;                  // position, also color (r,g,b)
    Vec(double x_=0, double y_=0, double z_=0){ x=x_; y=y_; z=z_; }
    Vec operator+(const Vec &b) const { return Vec(x+b.x,y+b.y,z+b.z); }
    Vec operator-(const Vec &b) const { return Vec(x-b.x,y-b.y,z-b.z); }
    Vec operator*(double b) const { return Vec(x*b,y*b,z*b); }
    Vec mult(const Vec &b) const { return Vec(x*b.x,y*b.y,z*b.z); }
    Vec& norm(){ return *this = *this * (1/sqrt(x*x+y*y+z*z)); }
    double dot(const Vec &b) const { return x*b.x+y*b.y+z*b.z; } // cross:
    Vec operator%(Vec&b){return Vec(y*b.z-z*b.y,z*b.x-x*b.z,x*b.y-y*b.x);}
};
struct Ray {
    Vec o, d;
    Ray(Vec o_, Vec d_) : o(o_), d(d_) {}
};
struct Sphere {
    double rad;       // radius
    Vec p, e, c;      // position, emission, color
    Refl_t refl;      // reflection type (DIFFuse, SPECular, REFRactive)
    Sphere(double rad_, Vec p_, Vec e_, Vec c_, Refl_t refl_):
    rad(rad_), p(p_), e(e_), c(c_), refl(refl_) {}
    double intersect(const Ray &r) const { // returns distance, 0 if nohit
        Vec op = p-r.o; // Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0
        double t, eps=1e-4, b=op.dot(r.d), det=b*b-op.dot(op)+rad*rad;
        if (det<0)
            return 0;
        else
            det=sqrt(det);
        return (t=b-det)>eps ? t : ((t=b+det)>eps ? t : 0);
    }
};
Sphere spheres[] = {
    //Scene: radius, position,         emission,        color,              material
    Sphere(1e5, Vec( 1e5+1,40.8,81.6), Vec(),           Vec(.75,.25,.25),   DIFF),//Left
    Sphere(1e5, Vec(-1e5+99,40.8,81.6),Vec(),           Vec(.25,.25,.75),   DIFF),//Rght
    Sphere(1e5, Vec(50,40.8, 1e5),     Vec(),           Vec(.75,.75,.75),   DIFF),//Back
//    Sphere(1e5, Vec(50,40.8,-1e5+170), Vec(),           Vec(),              DIFF),//Frnt
    Sphere(1e5, Vec(50, 1e5, 81.6),    Vec(),           Vec(.75,.75,.75),   DIFF),//Botm
    Sphere(1e5, Vec(50,-1e5+81.6,81.6),Vec(),           Vec(.75,.75,.75),   DIFF),//Top
    Sphere(16.5,Vec(27,16.5,47),       Vec(),           Vec(1,1,1)*.999,    SPEC),//Mirr
    Sphere(16.5,Vec(73,16.5,78),       Vec(),           Vec(1,1,1)*.999,    REFR),//Glas
    Sphere(600, Vec(50,681.6-.27,81.6),Vec(12,12,12),   Vec(),              DIFF) //Lite
};
inline double clamp(double x){ return x<0 ? 0 : x>1 ? 1 : x; }
inline int toInt(double x){ return int(pow(clamp(x),1/2.2)*255+.5); }


inline bool intersect(const Ray &r, double &t, int &id){
    double n=sizeof(spheres)/sizeof(Sphere), d, inf=t=1e20;
//    for(int i=int(n);i--;)
    for(int i=0; i<int(n);i++)
        if((d=spheres[i].intersect(r))&&d<t){
            t=d;id=i;
        }
    return t<inf;
}
Vec radiance(const Ray &r, int depth, unsigned short *Xi){// xi used to store the result of erand48 *******
    if (depth > 1) {
        return Vec();
    }
    double t;                               // distance to intersection
    int id=0;                               // id of intersected object
    if (!intersect(r, t, id))
        return Vec(); // if miss, return black
    const Sphere &obj = spheres[id];        // the hit object
    Vec x=r.o+r.d*t;// 射线与球的交点坐标
    Vec n=(x-obj.p).norm();// 球心到交点的向量
    Vec nl=n.dot(r.d)<0?n:n*-1;// 交点在正面还是背面，nl是反射面的法线
    Vec f=obj.c;// 球的颜色
    double p = f.x>f.y && f.x>f.z ? f.x : f.y>f.z ? f.y : f.z; // max refl，这儿为什么要选值最大的颜色呀？
    if (++depth>5) {
//        return obj.e;// 改了这一行，好像也没啥不一样呀～
        if (erand48(Xi)<p)
            f=f*(1/p);
        else
            return obj.e; //R.R.
    }
    if (obj.refl == DIFF){                  // Ideal DIFFUSE reflection
        double r1=2*M_PI*erand48(Xi);
        double r2=erand48(Xi);
        double r2s=sqrt(r2);
        Vec w=nl;// 反射时的法线
        Vec u=((fabs(w.x)>.1?Vec(0,1):Vec(1))%w).norm();
        Vec v=w%u;
        Vec d = (u*cos(r1)*r2s + v*sin(r1)*r2s + w*sqrt(1-r2)).norm();// 这儿的d为什么要这么计算？应该是一个随机的方向，如果这个方向对视点是不可见的呢？
        return obj.e + f.mult(radiance(Ray(x,d),depth,Xi));
    } else if (obj.refl == SPEC)            // Ideal SPECULAR reflection
        return obj.e + f.mult(radiance(Ray(x,r.d-n*2*n.dot(r.d)),depth,Xi));// 球面的反射方向
    Ray reflRay(x, r.d-n*2*n.dot(r.d));     // Ideal dielectric REFRACTION
    bool into = n.dot(nl)>0;                // Ray from outside going in?
    double nc=1, nt=1.5;
    double nnt=into?nc/nt:nt/nc;            // 这儿是折射率吗？
    double ddn=r.d.dot(nl), cos2t;
    if ((cos2t=1-nnt*nnt*(1-ddn*ddn))<0)    // Total internal reflection 这儿是计算内部的反射,全反射
        return obj.e + f.mult(radiance(reflRay,depth,Xi));
    Vec tdir = (r.d*nnt - n*((into?1:-1)*(ddn*nnt+sqrt(cos2t)))).norm(); // 这个应该是折射方向
    double a=nt-nc; // 0.5
    double b=nt+nc; // 2.5
    double R0=a*a/(b*b); // 0.25 / 6.25
    double c = 1-(into?-ddn:tdir.dot(n));
    double Re=R0+(1-R0)*c*c*c*c*c;
    double Tr=1-Re;
    double P=.25+.5*Re;
    double RP=Re/P;
    double TP=Tr/(1-P);
    // Russian roulette
    return obj.e + f.mult(depth>2 ?
                          (erand48(Xi)<P ? radiance(reflRay,depth,Xi)*RP:radiance(Ray(x,tdir),depth,Xi)*TP) :
                                           radiance(reflRay,depth,Xi)*Re+radiance(Ray(x,tdir),depth,Xi)*Tr);
}

Scene gScene;
void initScene() {
    Object* obj = nullptr;
    obj = new CSphere(1e5, vec3d( 1e5+1,40.8,81.6), vec3d(),           vec3d(.75,.25,.25),   DIFF);
    obj->setEntityName("1");
    gScene.addObject(obj);
    obj = new CSphere(1e5, vec3d(-1e5+99,40.8,81.6),vec3d(),           vec3d(.25,.25,.75),   DIFF);
    obj->setEntityName("2");
    gScene.addObject(obj);
    obj = new CSphere(1e5, vec3d(50,40.8, 1e5),     vec3d(),           vec3d(.75,.75,.75),   DIFF);
    obj->setEntityName("3");
    gScene.addObject(obj);
//    obj = new CSphere(1e5, vec3d(50,40.8,-1e5+170), vec3d(),           vec3d(),              DIFF);
//    obj->setEntityName("4");
//    gScene.addObject(obj);
    obj = new CSphere(1e5, vec3d(50, 1e5, 81.6),    vec3d(),           vec3d(.75,.75,.75),   DIFF);
    obj->setEntityName("5");
    gScene.addObject(obj);
    obj = new CSphere(1e5, vec3d(50,-1e5+81.6,81.6),vec3d(),           vec3d(.75,.75,.75),   DIFF);
    obj->setEntityName("6");
    gScene.addObject(obj);
    obj = new CSphere(16.5,vec3d(27,16.5,47),       vec3d(),           vec3d(1,1,1)*.999,    SPEC);
    obj->setEntityName("7");
    gScene.addObject(obj);
    obj = new CSphere(16.5,vec3d(73,16.5,78),       vec3d(),           vec3d(1,1,1)*.999,    REFR);
    obj->setEntityName("8");
    gScene.addObject(obj);
    obj = new CSphere(600, vec3d(50,681.6-.27,81.6),vec3d(12,12,12),   vec3d(),              DIFF);
    obj->setEntityName("9");
    gScene.addObject(obj);
    
//    gScene.addObject(  );//Left
//    gScene.addObject(  );//Rght
//    gScene.addObject(  );//Back
//    gScene.addObject(  );//Frnt
//    gScene.addObject(  );//Botm
//    gScene.addObject(  );//Top
//    gScene.addObject(  );//Mirr
//    gScene.addObject(  );//Glas
//    gScene.addObject(  );//Lite
}
void trace(int sample_count){
//int main(int argc, char *argv[]){
    int w=1024, h=768, samps = sample_count/4; // # samples
    Ray cam(Vec(50,52,295.6), Vec(0, 0,-1).norm()); // cam pos, dir -0.042612
    const float rate = 0.5153f; // 0.5153f; // 这个好像是FOV，是y方向上的角度
    Vec cx=Vec(w*rate/h); // 这个cx是干什么的，为什么要乘以0.5135
    printf("cx %f, %f %f", cx.x, cx.y, cx.z);
    Vec cy=(cx%cam.d).norm()*rate;// cy的最大值就是rate
    Vec *c=new Vec[w*h];
    Vec r;
    initScene();
    //char * dir = getcwd(NULL, 0);
#pragma omp parallel for schedule(dynamic, 1) private(r)       // OpenMP
    for (int y=0; y<h; y++){                       // Loop over image rows
        fprintf(stderr,"\rRendering (%d spp) %5.2f%% cx:%f %f %f",samps*4,100.*y/(h-1), cx.x, cx.y, cx.z);
        for (unsigned short x=0, Xi[3]={0,0,static_cast<unsigned short>(y*y*y)}; x<w; x++)   // Loop cols
        {
            // 每个点在2x2的大小上计算
            for (int sy=0, i=(h-y-1)*w+x; sy<2; sy++)     // 2x2 subpixel rows
            {
                for (int sx=0; sx<2; sx++, r=Vec()){        // 2x2 subpixel cols
                    // samps是输入的参数再除以4
                    vec3d tmpRes;
                    for (int s=0; s<samps; s++){
                        // dx, dy是什么，为什么要与随机数搞到一块儿？
                        double r1=2*erand48(Xi);
                        double dx=r1<1 ? sqrt(r1)-1: 1-sqrt(2-r1);
                        double r2=2*erand48(Xi);
                        double dy=r2<1 ? sqrt(r2)-1: 1-sqrt(2-r2);
                        // 下面是计算光线的方向
                        // 近裁剪面的x是从-0.5到0.5吗？
                        // 近裁剪面到相机的距离是1
                        Vec d = cx*( ( (sx+.5 + dx)/2 + x)/w - 0.5) + cy*( ( (sy+.5 + dy)/2 + y)/h - .5) + cam.d;
                        d.norm();
                        
#ifdef USE_XW
                        {
                            vec3d ori(cam.o.x, cam.o.y, cam.o.z);
                            vec3d dir(d.x, d.y, d.z);
                            dir = glm::normalize(dir);
                            
                            TRay ray(ori, dir);
                            vec3d ret = gScene.radiance(ray, 0, Xi) * (1.0/samps);
                            tmpRes += ret;
                        }
#else

                        // 下面的光线是什么意思？相机的位置不应该改变呀～
                        r = r + radiance(Ray(cam.o,d.norm()),0,Xi) * (1.0/samps);// 这儿也做了一次除法，避免一直计算时，最后的颜色成了白色
                        // 下面的140相当于近裁剪面，去掉貌似没什么影响
//                        r = r + radiance(Ray(cam.o+d*140,d.norm()),0,Xi)*(1./samps);// 这儿也做了一次除法，避免一直计算时，最后的颜色成了白色
#endif
                    } // Camera rays are pushed ^^^^^ forward to start in interior
#ifdef USE_XW
                    Vec pointValue = Vec(clamp(tmpRes.x),clamp(tmpRes.y),clamp(tmpRes.z))*.25;//这儿除以4
#else
                    Vec pointValue = Vec(clamp(r.x),clamp(r.y),clamp(r.z))*.25;//这儿除以4
#endif
                    pointValue = pointValue + c[i];
                    c[i] = pointValue;
                }
            }
//            fprintf(stderr, "%f %f %f", c[i].x, c[i].y, c[i].z);
        }
    }
    FILE *f = fopen("/Users/xiewei/Desktop/image.ppm", "w");         // Write image to PPM file.
    fprintf(f, "P3\n%d %d\n%d\n", w, h, 255);
    for (int i=0; i<w*h; i++) {
        fprintf(f,"%d %d %d ", toInt(c[i].x), toInt(c[i].y), toInt(c[i].z));
    }
    fclose(f);
    return;
}
