// console.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
extern void trace(int sampleCount, const char* dir, bool isDevice);

int main()
{
    trace(30, "./", true);
    return 0;
}

