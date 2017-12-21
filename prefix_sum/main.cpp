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
#include <algorithm>
#include <functional>
#include <cstdio>
#include <cmath>
#include <cassert>
#include <numeric>
#include <complex>

using uint = size_t;

int closest_power_of_two(const int x)
{
    if (x == 0)
    {
        return 0;
    }

    int i = 0;
    int tmp = x;
    do
    {
        i++;
        tmp >>= 1;
    } while (tmp);


    if (1 << (i - 1) == x)
    {
        return x;
    }
    return 1 << i;
}

#define xstr(s) str(s)
#define str(s) #s
#define SCAN_BLOCK_SIZE 256
#define TYPE double
#define TYPE_SIZE sizeof(TYPE)

using vector = std::vector<TYPE>;

void profile_events(const cl::Event &start_event, const cl::Event finish_event, const std::string &s = "Total time: ")
{
    cl_ulong start_time = start_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    cl_ulong end_time = finish_event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
    cl_ulong elapsed_time = end_time - start_time;
    std::cout << std::setprecision(2) << s << elapsed_time / 1000000.0 << " ms" << std::endl;
}

vector solve(const vector &v)
{
    const int n = v.size();

    std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;
    std::vector<cl::Kernel> kernels;

    cl::Platform::get(&platforms);
    platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

    cl::Context context(devices);
    cl::CommandQueue queue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);

    std::ifstream cl_file("prefix_sum.cl");
    std::string cl_string(std::istreambuf_iterator<char>(cl_file), (std::istreambuf_iterator<char>()));

    cl::Program::Sources source({ cl_string });
    cl::Program program(context, source);
    try
    {
        const std::string build_str = "-D BLOCK_SIZE=" + std::to_string(SCAN_BLOCK_SIZE);
        program.build(devices, build_str.c_str());
    }
    catch (cl::Error &ignored)
    {
        std::string log;
        program.getBuildInfo(devices[0], CL_PROGRAM_BUILD_LOG, &log);
        std::cout << log << std::endl;
        throw;
    }

    const int N = std::max(SCAN_BLOCK_SIZE, closest_power_of_two(n));

    cl::Buffer dev_input(context, CL_MEM_READ_ONLY, TYPE_SIZE * N);
    cl::Buffer dev_output(context, CL_MEM_READ_WRITE, TYPE_SIZE * N);

    auto a = cl::Local(TYPE_SIZE * SCAN_BLOCK_SIZE);
    auto b = cl::Local(TYPE_SIZE * SCAN_BLOCK_SIZE);

    queue.enqueueWriteBuffer(dev_input, CL_TRUE, 0, TYPE_SIZE * v.size(), v.data());

    cl::EnqueueArgs global_args{ queue, cl::NullRange, cl::NDRange(N), cl::NDRange(SCAN_BLOCK_SIZE) };

    auto bottom_up = cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::LocalSpaceArg, cl::LocalSpaceArg>(program, "bottom_up");
    auto start_event = bottom_up(global_args, dev_input, dev_output, a, b);
    start_event.wait();
    profile_events(start_event, start_event, "bottom_up: ");

    auto reduce_blocks = cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::LocalSpaceArg, cl::LocalSpaceArg, uint>(program, "reduce_blocks");
    const int NBLOCKS = std::max(N / SCAN_BLOCK_SIZE, SCAN_BLOCK_SIZE);
    cl::Buffer block_output(context, CL_MEM_READ_WRITE, TYPE_SIZE * NBLOCKS);
    cl::EnqueueArgs block_args{ queue, cl::NullRange, cl::NDRange(SCAN_BLOCK_SIZE), cl::NDRange(SCAN_BLOCK_SIZE) };
    auto event = reduce_blocks(block_args, dev_output, block_output, a, b, N);
    event.wait();
    profile_events(event, event, "reduce_blocks: ");

    auto reduce_all = cl::KernelFunctor<cl::Buffer, cl::Buffer>(program, "reduce_all");
    auto finish_event = reduce_all(global_args, dev_output, block_output);
    finish_event.wait();
    profile_events(finish_event, finish_event, "bottom_up: ");

    vector result(v.size());
    queue.enqueueReadBuffer(dev_output, CL_TRUE, 0, TYPE_SIZE * v.size(), result.data());
    queue.finish();
    return result;
}

int main()
{
    std::ifstream fin("input.txt");
    int n;
    fin >> n;
    vector v(n);
    for (int i = 0; i < n; ++i)
    {
        fin >> v[i];
    }

    auto result = solve(v);
    bool test_run = false;
    if (!test_run)
    {
        std::ofstream fout("output.txt");
        fout << std::fixed << std::setprecision(3);
        std::copy(result.begin(), result.end(), std::ostream_iterator<TYPE>(fout, " "));
    }
    else {
        try
        {
            //std::cout << "Result: " << std::endl;
            //std::copy(result.begin(), result.end(), std::ostream_iterator<double>(std::cout, " "));
            //std::cout << std::endl;
            vector expected;
            std::partial_sum(v.begin(), v.end(), std::back_inserter(expected));
            if (result != expected)
            {
                double s = 0.0;
                for (int i = 0; i < n; ++i)
                {
                    if (i > 0 && result[i] != result[i - 1] + 1)
                    {
                        std::cout << i + 1 << std::endl;
                        break;
                    }
                    s += std::abs(result[i] - expected[i]);
                }

                std::cout << "Error: " << std::setprecision(2) << s << std::endl;
            }
        }
        catch (cl::Error &e)
        {
            std::cout << "[OpenCL ERROR] " << e.what() << " : " << e.err() << std::endl;
        }
        catch (std::exception &e)
        {
            std::cout << "[ERROR] " << e.what();
        }
    }

    return 0;
}