#define _CRT_SECURE_NO_WARNINGS
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/cl.h>
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#include "cl2.hpp"

#include <vector>
#include <fstream>
#include <iostream>
#include <iterator>
#include <iomanip>

int main()
{
    std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;
    std::vector<cl::Kernel> kernels;

    try {
        // create platform
        cl::Platform::get(&platforms);
        platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

        // create context
        cl::Context context(devices);

        // create command queue
        cl::CommandQueue queue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);

        // load opencl source
        std::ifstream cl_file("scan.cl");
        std::string cl_string(std::istreambuf_iterator<char>(cl_file), (std::istreambuf_iterator<char>()));
        cl::Program::Sources source({ cl_string });

        // create program
        cl::Program program(context, source);

        // compile opencl source
        program.build(devices);

        // create a message to send to kernel
        size_t const block_size = 512;
        size_t const test_array_size = 512;
        size_t const output_size = test_array_size;
        std::vector<int> input(test_array_size);
        std::vector<int> output(output_size, 0);
        for (size_t i = 0; i < test_array_size; ++i)
        {
            input[i] = i % 10;
        }

        // allocate device buffer to hold message
        cl::Buffer dev_input (context, CL_MEM_READ_ONLY, sizeof(int) * test_array_size);
        cl::Buffer dev_output(context, CL_MEM_WRITE_ONLY, sizeof(int) * output_size);

        // copy from cpu to gpu
        queue.enqueueWriteBuffer(dev_input, CL_TRUE, 0, sizeof(int) * test_array_size, &input[0]);

        queue.finish();

        // load named kernel from opencl source
        auto scan_hs = cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::LocalSpaceArg, cl::LocalSpaceArg>(program, "scan_hillis_steele");
        cl::EnqueueArgs scan_hs_args{ queue, cl::NullRange, cl::NDRange(test_array_size), cl::NDRange(block_size) };
        cl::Event event = scan_hs(scan_hs_args, dev_input, dev_output, cl::Local(sizeof(int) * block_size), cl::Local(sizeof(int) * block_size));

        //auto scan_b = cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::LocalSpaceArg>(program, "scan_blelloch");
        //cl::EnqueueArgs scan_b_args{ queue, cl::NullRange, cl::NDRange(test_array_size), cl::NDRange(block_size) };
        //cl::Event event = scan_b(scan_b_args, dev_input, dev_output, cl::Local(sizeof(int) * block_size));

        event.wait();
        cl_ulong start_time = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
        cl_ulong end_time    = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
        cl_ulong elapsed_time = end_time - start_time;

        queue.enqueueReadBuffer(dev_output, CL_TRUE, 0, sizeof(int) * output_size, &output[0]);

        std::cout << std::setprecision(2) << "Total time: " << elapsed_time / 1000000.0 << " ms" << std::endl;

    }
    catch (cl::Error e)
    {
        std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
    }

    return 0;
}