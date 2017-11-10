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
#include <cmath>
#include <cassert>

//Проверьте, что ваша программа работает на следующих примерах :
//N = 1024. M = 3. Обе матрицы состоят из единиц.
//N = 1024. M = 9. Обе матрицы состоят из единиц.
//N = 1. M = 9. Обе матрицы состоят из единиц.
//N = 31. M = 9. Обе матрицы состоят из единиц.
//N = 1023. M = 9. Обе матрицы состоят из единиц.

template<typename T, typename Comp=std::equal_to<T>>
struct MatrixT
{
    MatrixT(int n)
        : n_(n)
        , v_(n * n, 0)
    {}
    
    T &get(int i, int j)
    {
        return v_[index(i, j)];
    }

    const T &get(int i, int j) const
    {
        return v_[index(i, j)];
    }

    int dim() const
    {
        return n_;
    }

    int size() const
    {
        return n_ * n_;
    }

    T *data()
    {
        return &v_[0];
    }

    friend std::istream &operator >> (std::istream &is, MatrixT &matrix) {
        for (auto & x : matrix.v_)
        {
            is >> x;
        }
        return is;
    }

    friend std::ostream &operator << (std::ostream &os, const MatrixT &matrix) {
        for (int i = 0; i < matrix.n_; ++i)
        {
            for (int j = 0; j < matrix.n_; ++j)
            {
                os << matrix.get(i, j) << ' ';
            }
            os << std::endl;
        }
        return os;
    }

    friend bool operator==(const MatrixT &lhs, const MatrixT &rhs)
    {
        if (lhs.n_ != rhs.n_)
        {
            return false;
        }
        const size_t n = lhs.n_;
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                if (!comparator_(lhs.get(i, j), rhs.get(i, j)))
                {
                    return false;
                }
            }
        }
        return true;
    }

private:
    static Comp comparator_;
    int n_;
    std::vector<T> v_;

    size_t index(int i, int j) const
    {
        const int bias = i * n_ + j;
        assert(bias >= 0 && bias < size());
        return static_cast<size_t>(bias);
    }
};

struct DoubleComparator
{
    static const double EPS;

    bool operator()(double a, double b) const {
        return fabs(a - b) < EPS;
    }
};
const double DoubleComparator::EPS = 1e-6;

typedef MatrixT<double, DoubleComparator> Matrix;

Matrix convolve_2d_cpu(const Matrix &a, const Matrix &b)
{
    const int n = a.dim();
    const int m = b.dim();
    assert(m % 2 != 0);
    const int hm = (m - 1) / 2;

    Matrix res{n};

    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            for (int k = -hm; k <= hm; ++k)
            {
                if (i + k < 0 || i + k >= n) continue;
                for (int l = -hm; l <= hm; ++l)
                {
                    if (j + l < 0 || j + l >= n) continue;
                    res.get(i, j) += a.get(i + k, j + l) * b.get(k + hm, l + hm);
                }
            }
        }
    }
    return res;
}

int main()
{
    using std::vector;
    using std::string;
    using std::ifstream;
    using std::cout;
    using std::endl;

    ifstream fin("input.txt");
    int n, m;
    fin >> n >> m;
    Matrix a(n), b(m);
    fin >> a >> b;
    cout << convolve_2d_cpu(a, b) << endl;

    vector<cl::Platform> platforms;
    vector<cl::Device> devices;
    vector<cl::Kernel> kernels;

    try {
        // create platform
        cl::Platform::get(&platforms);
        platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

        // create context
        cl::Context context(devices);

        // create command queue
        cl::CommandQueue queue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);

        // load opencl source
        ifstream cl_file("convolution_2d.cl");
        string cl_string(std::istreambuf_iterator<char>(cl_file), (std::istreambuf_iterator<char>()));

        cl::Program::Sources source({ cl_string });

        // create program
        cl::Program program(context, source);

        // compile opencl source
        program.build(devices);

        // create a message to send to kernel
        size_t const block_size = 512;
        size_t const test_array_size = 512;
        size_t const mask_size = 9;

        vector<int> input(test_array_size, 1);
        vector<int> output(test_array_size, 1);
        int mask[mask_size] = { 1, 1, 1, 1, -1, 1, 1, 1, 1 };

        // allocate device buffer to hold message
        cl::Buffer dev_input(context, CL_MEM_READ_ONLY, sizeof(int) * test_array_size);
        cl::Buffer dev_output(context, CL_MEM_WRITE_ONLY, sizeof(int) * test_array_size);
        cl::Buffer dev_mask(context, CL_MEM_READ_ONLY, sizeof(int) * mask_size);

        // copy from cpu to gpu
        queue.enqueueWriteBuffer(dev_input, CL_TRUE, 0, sizeof(int) * test_array_size, &input[0]);
        queue.enqueueWriteBuffer(dev_mask, CL_TRUE, 0, sizeof(int)* mask_size, &mask[0]);

        // load named kernel from opencl source
        queue.finish();
        cl::Kernel kernel_gmem(program, "gpu_convolution_2d_gmem");
        auto convolution_gmem = cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, int, int>(kernel_gmem);
        cl::EnqueueArgs convolution_gmem_args{ queue, cl::NullRange, cl::NDRange(test_array_size), cl::NDRange(block_size) };
        cl::Event event = convolution_gmem(convolution_gmem_args, dev_input, dev_mask, dev_output, mask_size, test_array_size);
        //cl::KernelFunctor convolution_gmem(kernel_gmem, queue, cl::NullRange, cl::NDRange(test_array_size), cl::NDRange(block_size));
        //cl::Event event = convolution_gmem(dev_input, dev_mask, dev_output, mask_size, test_array_size);

        //cl::Kernel kernel_lmem(program, "gpu_convolution_lmem");
        //auto convolution_lmem = cl::KernelFunctor<>(kernel_lmem);
        //cl::EnqueueArgs convolution_lmem_args{ cl::NullRange, cl::NDRange(test_array_size), cl::NDRange(block_size) };
        //convolution_lmem(convolution_lmem_args);
        //cl::KernelFunctor convolution_lmem(kernel_lmem, queue, cl::NullRange, cl::NDRange(test_array_size), cl::NDRange(block_size));
        //cl::Event event = convolution_lmem(dev_input, dev_mask, dev_output, mask_size, test_array_size, 
        //cl::__local((block_size + mask_size) * sizeof(int)));

        event.wait();
        cl_ulong start_time = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
        cl_ulong end_time = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
        cl_ulong elapsed_time = end_time - start_time;

        queue.enqueueReadBuffer(dev_output, CL_TRUE, 0, sizeof(int) * test_array_size, &output[0]);
        for (size_t i = 0; i < test_array_size; ++i)
            cout << output[i] << " ";
        cout << endl;

        cout << std::setprecision(2) << "Total time: " << elapsed_time / 1000000.0 << " ms" << endl;
    }
    catch (cl::Error e)
    {
        cout << "[ERROR] " << e.what() << " : " << e.err() << endl;
    }

    return 0;
}