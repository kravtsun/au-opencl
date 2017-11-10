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

//Проверьте, что ваша программа работает на следующих примерах :
//N = 1024. M = 3. Обе матрицы состоят из единиц.
//N = 1024. M = 9. Обе матрицы состоят из единиц.
//N = 1. M = 9. Обе матрицы состоят из единиц.
//N = 31. M = 9. Обе матрицы состоят из единиц.
//N = 1023. M = 9. Обе матрицы состоят из единиц.


#define DOUBLE_EPS 1e-6
struct Matrix
{
    Matrix(int n)
        : n_(n)
        , v_(n * n, 0)
    {}
    
    double &get(int i, int j)
    {
        return v_[index(i, j)];
    }

    const double &get(int i, int j) const
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

    double *data()
    {
        return &v_[0];
    }

    friend std::istream &operator >> (std::istream &is, Matrix &matrix) {
        for (auto & x : matrix.v_)
        {
            is >> x;
        }
        return is;
    }

    friend std::ostream &operator << (std::ostream &os, const Matrix &matrix) {
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

    static bool are_equal(double lhs, double rhs)
    {
        return fabs(lhs - rhs) < DOUBLE_EPS;
    }

    friend bool operator==(const Matrix &lhs, const Matrix &rhs)
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
                if (!are_equal(lhs.get(i, j), rhs.get(i, j)))
                {
                    return false;
                }
            }
        }
        return true;
    }

    friend bool operator!=(const Matrix &lhs, const Matrix &rhs)
    {
        return !(lhs == rhs);
    }

private:
    int n_;
    std::vector<double> v_;

    size_t index(int i, int j) const
    {
        const int bias = i * n_ + j;
        assert(bias >= 0 && bias < size());
        return static_cast<size_t>(bias);
    }
};

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


#define STRINGIFY(x) #x
#define BLOCK_SIZE 16

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
    auto expected = convolve_2d_cpu(a, b);

    vector<cl::Platform> platforms;
    vector<cl::Device> devices;
    vector<cl::Kernel> kernels;

    try {
        cl::Platform::get(&platforms);
        platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

        cl::Context context(devices);
        cl::CommandQueue queue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);

        ifstream cl_file("convolution_2d.cl");
        string cl_string(std::istreambuf_iterator<char>(cl_file), (std::istreambuf_iterator<char>()));

        cl::Program::Sources source({ cl_string });
        cl::Program program(context, source);
        program.build(devices, "-D BLOCK_SIZE=" STRINGIFY(BLOCK_SIZE));
        const size_t double_size = sizeof(double);
        Matrix result{ n };
        cl::Buffer dev_input(context, CL_MEM_READ_ONLY, double_size * a.size());
        cl::Buffer dev_mask(context, CL_MEM_READ_ONLY, double_size * b.size());
        cl::Buffer dev_output(context, CL_MEM_WRITE_ONLY, double_size * result.size());

        // copy from cpu to gpu
        queue.enqueueWriteBuffer(dev_input, CL_TRUE, 0, double_size * a.size(), a.data());
        queue.enqueueWriteBuffer(dev_mask, CL_TRUE, 0, double_size * b.size(), b.data());

        auto convolution_gmem = cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, int, int>(program, "gpu_convolution_2d_gmem");
        const int N = std::max(32, closest_power_of_two(n));
        cl::EnqueueArgs convolution_gmem_args{ queue, cl::NullRange, cl::NDRange(N, N), cl::NDRange(BLOCK_SIZE)};
        cl::Event event = convolution_gmem(convolution_gmem_args, dev_input, dev_mask, dev_output, m, n);
        event.wait();        
        queue.enqueueReadBuffer(dev_output, CL_TRUE, 0, double_size * result.size(), result.data());

        if (result != expected)
        {
            for (int i = 0; i < n; ++i)
            {
                for (int j = 0; j < n; ++j)
                {
                    if (!Matrix::are_equal(result.get(i, j), expected.get(i, j)))
                    {
                        const auto index_string = std::to_string(i) + ", " + std::to_string(j);
                        throw std::logic_error("result is not equal to expected, differs in " + index_string + "\n");
                    }
                }
            }
            assert(false);
        }

        FILE *fout = fopen("output.txt", "w");

        if (fout == nullptr)
        {
            throw "Unable to open file";
        }

        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                fprintf(fout, "%.3lf ", result.get(i, j));
            }
            fprintf(fout, "\n");
        }
        fclose(fout);
    }
    catch (cl::Error &e)
    {
        cout << "[OpenCL ERROR] " << e.what() << " : " << e.err() << endl;
    }
    catch (std::exception &e)
    {
        cout << "[ERROR] " << e.what();
    }

    return 0;
}