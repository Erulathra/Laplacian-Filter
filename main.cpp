#include <iostream>
#include <memory>
#include <fstream>

#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#include "spdlog/spdlog.h"

#define __CL_ENABLE_EXCEPTIONS

#if __APPLE__
#include <OpenCL/opencl.h>
#else

#include <CL/cl.hpp>

#endif

cl::Device GetGPUDevice();
cl::Kernel LoadKernel(const std::string& path, const std::string& name, const cl::Device& device, const cl::Context& context);
std::vector<float> LoadFilter(const std::string& path);

int main(int argc, char** argv) {

    // Load Image
    cv::Mat image = cv::imread("res/image.jpg", cv::IMREAD_UNCHANGED);
    image.convertTo(image, CV_8UC4);
    cv::cvtColor(image, image, cv::ColorConversionCodes::COLOR_BGR2RGBA);

    SPDLOG_INFO(image.type() == CV_8UC4);


    if (image.empty())
        return -1;

    // Get device
    cl::Device gpuDevice = GetGPUDevice();
    std::cout << gpuDevice.getInfo<CL_DEVICE_NAME>() << std::endl;

    cl::Context context{gpuDevice};
    cl::CommandQueue commandQueue{context, gpuDevice};

    cl_int status = 0;
    cl::Image2D source(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, cl::ImageFormat(CL_RGBA, CL_UNSIGNED_INT8),
                       image.cols, image.rows, 0, image.data);
    cl::Image2D destination(context, CL_MEM_WRITE_ONLY, cl::ImageFormat(CL_RGBA, CL_UNSIGNED_INT8), image.cols, image.rows);

    std::vector<float> filterMatrix = LoadFilter("res/filter.txt");
    cl::Buffer filterBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, filterMatrix.size() * sizeof(float),
                      (void*) &filterMatrix[0]);


    cl::Kernel laplacianKernel = LoadKernel("res/kernel.cl", "Laplacian", gpuDevice, context);
    laplacianKernel.setArg(0, source);
    laplacianKernel.setArg(1, destination);
    laplacianKernel.setArg(2, filterBuffer);


    // execute kernel
    commandQueue.enqueueNDRangeKernel(laplacianKernel, cl::NullRange, cl::NDRange(image.cols, image.rows));

    cv::Mat result(image.rows, image.cols, CV_8UC4);
    SPDLOG_INFO("Result Size: {}x{}", result.cols, result.rows);

    cl::size_t<3> origin;
    origin[0] = 0;
    origin[1] = 0;
    origin[2] = 0;
    cl::size_t<3> region;
    region[0] = result.cols;
    region[1] = result.rows;
    region[2] = 1;

    commandQueue.enqueueReadImage(destination, CL_TRUE, origin, region, 0, 0, result.data);
    cv::cvtColor(result, result, cv::ColorConversionCodes::COLOR_RGBA2BGR);
    cv::imwrite("res/output.png", result);

    return 0;
}

cl::Device GetGPUDevice() {

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    if (platforms.empty()) {
        SPDLOG_ERROR("No platforms found!");
        exit(1);
    }

    /**
     * Search for all the devices on the first platform
     * and check if there are any available.
     * */

    auto platform = platforms.front();
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);

    if (devices.empty()) {
        SPDLOG_ERROR("No GPU found!");
        exit(1);
    }

    /**
     * Return the first device found.
     * */

    return devices.front();
}

cl::Kernel LoadKernel(const std::string& path, const std::string& name, const cl::Device& device, const cl::Context& context) {
    std::ifstream kernelFile{path};
    std::stringstream kernelStringStream;

    kernelStringStream << kernelFile.rdbuf();
    std::string kernelSource = kernelStringStream.str();

    kernelFile.close();

    cl::Program program(context, kernelSource, GL_TRUE);

    if(program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device) != 0){
        std::cerr << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device) << "/n"
                  << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
        exit(1);
    }

    cl::Kernel kernel{program, name.c_str(), nullptr};


    return kernel;
}

std::vector<float> LoadFilter(const std::string& path) {
    std::vector<float> result;

    std::ifstream filterFile{path};
    while (!filterFile.eof()) {
        float value;
        filterFile >> value;
        result.push_back(value);
    }

    return result;
}