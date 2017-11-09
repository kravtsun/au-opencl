#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/cl.h>
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#include "cl2.hpp"

#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <iterator>

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
        cl::CommandQueue queue(context, devices[0]);

        // load opencl source
        std::ifstream cl_file("matrix_mult.cl");
        std::string cl_string(std::istreambuf_iterator<char>(cl_file), (std::istreambuf_iterator<char>()));
        cl::Program::Sources source({cl_string});

        // create program
        cl::Program program(context, source);

        // compile opencl source
        size_t const block_size = 16;
        program.build(devices, "-D BLOCK_SIZE=16");

        // create a message to send to kernel
        size_t const N = 256;
        size_t const matrix_size = N * N;

        int a[matrix_size];
        int b[matrix_size];
        int c[matrix_size];
        for (size_t i = 0; i < N; ++i)
        {
            for (size_t j = 0; j < N; ++j)
            {
                size_t idx = i * N + j;
                a[idx] = rand() % 10;
                b[idx] = rand() % 10;
                c[idx] = 0;
            }
        }

        // allocate device buffer to hold message
        cl::Buffer dev_a(context, CL_MEM_READ_ONLY, sizeof(int) * matrix_size);
        cl::Buffer dev_b(context, CL_MEM_READ_ONLY, sizeof(int) * matrix_size);
        cl::Buffer dev_c(context, CL_MEM_WRITE_ONLY, sizeof(int) * matrix_size);

        // copy from cpu to gpu
        queue.enqueueWriteBuffer(dev_a, CL_TRUE, 0, sizeof(int) * matrix_size, a);
        queue.enqueueWriteBuffer(dev_b, CL_TRUE, 0, sizeof(int) * matrix_size, b);

        // load named kernel from opencl source
        cl::Kernel kernel(program, "matrix_mult");

        //auto convolution_gmem = cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, int, int>(kernel_gmem);
        //cl::EnqueueArgs convolution_gmem_args{ queue, cl::NullRange, cl::NDRange(test_array_size), cl::NDRange(block_size) };
        //cl::Event event = convolution_gmem(convolution_gmem_args, dev_input, dev_mask, dev_output, mask_size, test_array_size);
        auto matrix_mult = cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, int>(kernel);

        cl::EnqueueArgs enqueue_args{ queue, cl::NullRange, cl::NDRange(N, N), cl::NDRange(block_size, block_size) };
        auto event = matrix_mult(enqueue_args, dev_a, dev_b, dev_c, (int)N);
        //event.wait(); // ?

        //cl::Kernel kernel_shared(program, "matrix_mult_shared");
        //auto matrix_mult_shared = cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, int>(kernel_shared);
        //cl::EnqueueArgs shared_enqueue_args{ queue, cl::NullRange,
        //                                     cl::NDRange(N, N), cl::NDRange(block_size, block_size));
        //matrix_mult_shared(shared_enqueue_args, dev_a, dev_b, dev_c, (int)N);

        queue.enqueueReadBuffer(dev_c, CL_TRUE, 0, sizeof(int) * matrix_size, c);

        for (size_t i = 0; i < N; ++i)
        {
           for (size_t j = 0; j < N; ++j)
           {
              size_t idx = i * N + j;
              //std::cout << a[idx] << " ";
           }
           //std::cout << std::endl;
        }
        std::cout << std::endl;

        for (size_t i = 0; i < N; ++i)
        {
           for (size_t j = 0; j < N; ++j)
           {
              size_t idx = i * N + j;
              //std::cout << b[idx] << " ";
           }
           //std::cout << std::endl;
        }
        std::cout << std::endl;

        for (size_t i = 0; i < N; ++i)
        {
           for (size_t j = 0; j < N; ++j)
           {
              size_t idx = i * N + j;

              int sum = 0;
              for (int k = 0; k < N; ++k)
                 sum += a[i * N + k] * b[k * N + j];
              if (c[idx] != sum)
                  std::cout << i << " " << j << std::endl;
              //std::cout << c[idx] - sum << " ";
           }
           //std::cout << std::endl;
        }
        std::cout << "finished" << std::endl;
        }
            catch (cl::Error e)
        {
            std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
        }

        return 0;
    }