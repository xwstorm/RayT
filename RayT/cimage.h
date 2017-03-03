//
//  CImage.h
//  RayT
//
//  Created by xiewei on 17/3/2.
//  Copyright © 2017年 xiewei. All rights reserved.
//

#ifndef CImage_h
#define CImage_h

#include <stdio.h>
#include <string>
class CImage {
public:
    CImage();
    CImage(std::string& file);
    
    
protected:
    std::string mFileName;
};
#endif /* CImage_h */
