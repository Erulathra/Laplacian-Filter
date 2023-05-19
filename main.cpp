#include <iostream>
#include <memory>

#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"

#define __CL_ENABLE_EXCEPTIONS

#if __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

int main(int argc,char **argv) {
    cv::Mat image = cv::imread("res/image.jpg");


    cv::imwrite("res/output.png", image);

    return 0;
}