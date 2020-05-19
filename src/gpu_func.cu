#include "gpu_func.h"
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <thrust/scan.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/reverse_iterator.h>
#include <thrust/device_ptr.h>
#include <stdlib.h>

typedef thrust::device_vector<numeric>::iterator Iterator; 
typedef thrust::device_vector<int>::iterator   IndexIterator;
typedef thrust::permutation_iterator<Iterator, IndexIterator> PermIter;

#define THREAD_PER_BLOCK 128

void allocate_device_memory(cox_data &dev_data, cox_cache &dev_cache, cox_param &dev_param, uint32_t total_cases, uint32_t K, uint32_t p)
{
    cudaMalloc((void**)&dev_data.X, sizeof(numeric) *p * total_cases);
    cudaMalloc((void**)&dev_data.status, sizeof(numeric)  * total_cases*K);
    cudaMalloc((void**)&dev_data.rankmin, sizeof(int) * total_cases*K);
    cudaMalloc((void**)&dev_data.rankmax, sizeof(int) * total_cases*K);
    cudaMalloc((void**)&dev_data.order, sizeof(int) * total_cases*K);
    cudaMalloc((void**)&dev_data.rev_order, sizeof(int) * total_cases*K);

    cudaMalloc((void**)&dev_cache.outer_accumu, sizeof(numeric) * total_cases*K);
    cudaMalloc((void**)&dev_cache.eta, sizeof(numeric) * total_cases*K);
    cudaMalloc((void**)&dev_cache.exp_eta, sizeof(numeric) * total_cases*K);
    cudaMalloc((void**)&dev_cache.exp_accumu, sizeof(numeric) * total_cases*K);
    cudaMalloc((void**)&dev_cache.residual, sizeof(numeric) * total_cases*K);
    cudaMalloc((void**)&dev_cache.B_col_norm, sizeof(numeric) * p);
    cudaMalloc((void**)&dev_cache.cox_val, sizeof(numeric) * K);
    cudaMalloc((void**)&dev_cache.order_cache, sizeof(numeric) * total_cases*K);

    cudaMalloc((void**)&dev_param.B, sizeof(numeric) * K * p);
    cudaMalloc((void**)&dev_param.v, sizeof(numeric) * K * p);
    cudaMalloc((void**)&dev_param.grad, sizeof(numeric) * K * p);
    cudaMalloc((void**)&dev_param.prev_B, sizeof(numeric) * K * p);
    cudaMalloc((void**)&dev_param.grad_ls, sizeof(numeric) * K * p);
    cudaMalloc((void**)&dev_param.penalty_factor, sizeof(numeric) * p);
    cudaMalloc((void**)&dev_param.ls_result, sizeof(numeric) * 2);
    cudaMalloc((void**)&dev_param.change, sizeof(numeric) * 1);

}

void free_device_memory(cox_data &dev_data, cox_cache &dev_cache, cox_param &dev_param)
{
    cudaFree(dev_data.X);
    cudaFree(dev_data.status);
    cudaFree(dev_data.rankmax);
    cudaFree(dev_data.rankmin);

    cudaFree(dev_cache.outer_accumu);
    cudaFree(dev_cache.eta);
    cudaFree(dev_cache.exp_eta);
    cudaFree(dev_cache.exp_accumu);
    cudaFree(dev_cache.residual);
    cudaFree(dev_cache.B_col_norm);
    cudaFree(dev_cache.cox_val);

    cudaFree(dev_param.B);
    cudaFree(dev_param.v);
    cudaFree(dev_param.grad);
    cudaFree(dev_param.prev_B);
    cudaFree(dev_param.penalty_factor);
    cudaFree(dev_param.ls_result);
    cudaFree(dev_param.grad_ls);
    cudaFree(dev_param.change);

}

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

#endif


// double atomicMax, copied from https://github.com/treecode/Bonsai/blob/master/runtime/profiling/derived_atomic_functions.h
__device__ __forceinline__ double atomicMax(double *address, double val)
{
    unsigned long long ret = __double_as_longlong(*address);
    while(val > __longlong_as_double(ret))
    {
        unsigned long long old = ret;
        if((ret = atomicCAS((unsigned long long *)address, old, __double_as_longlong(val))) == old)
            break;
    }
    return __longlong_as_double(ret);
}

// float atomicMax
__device__ __forceinline__ float atomicMax(float *address, float val)
{
    int ret = __float_as_int(*address);
    while(val > __int_as_float(ret))
    {
        int old = ret;
        if((ret = atomicCAS((int *)address, old, __float_as_int(val))) == old)
            break;
    }
    return __int_as_float(ret);
}

// GPU reduction code, adapted from https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
// Of course we can do the same thing for Max, but we don't need it for now.
template<uint32_t blockSize>
__device__ void warpSum(volatile numeric *sdata, uint32_t tid){
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

template<uint32_t blockSize>
__device__ void reduce_sum(numeric *g_idata, numeric *g_odata, uint32_t n){
    __shared__ numeric sdata[THREAD_PER_BLOCK];
    uint32_t tid = threadIdx.x;
    uint32_t i = blockIdx.x*(blockSize*2) + tid;
    sdata[tid] = (i+blockSize<n)?g_idata[i+blockSize]:0.0;
    if(i < n){
        sdata[tid] += g_idata[i];
    }
    __syncthreads();
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
    if (tid < 32) warpSum<blockSize>(sdata, tid);
    if(tid == 0){
        atomicAdd(g_odata,sdata[0]);
    }

}


void compute_product(numeric *A, numeric *B, numeric *C, 
    uint32_t N, uint32_t p, uint32_t K, cudaStream_t stream, cublasHandle_t handle, cublasOperation_t trans=CUBLAS_OP_N)
{
    numeric alpha = 1.0;
    numeric beta = 0.0;
    cublasSetStream(handle, stream);
    int lda = (trans == CUBLAS_OP_N)?N:p;
    cublasSgemm(handle, trans, CUBLAS_OP_N, 
                N, K, p, &alpha, 
                A, lda, B, p, &beta, C, N);
}

__global__
void apply_exp_gpu(const numeric *x, numeric *ex, uint32_t len)
{
    uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid < len)
    {
        ex[tid] = exp(x[tid]);
    }
}

void apply_exp(const numeric *x, numeric *ex, uint32_t len, cudaStream_t stream)
{
    constexpr int num_thread = THREAD_PER_BLOCK;
    int num_block = (len + num_thread - 1)/num_thread;
    apply_exp_gpu<<<num_block, num_thread, 0, stream>>>(x, ex, len);
}

// do rev_cumsum of x and save it to y
void rev_cumsum(numeric *x, numeric *y, uint32_t len, cudaStream_t stream)
{
    thrust::device_ptr<numeric> dptr_x = thrust::device_pointer_cast<numeric>(x);
    thrust::reverse_iterator<Iterator> r_x = make_reverse_iterator(dptr_x+len);
    thrust::device_ptr<numeric> dptr_y = thrust::device_pointer_cast<numeric>(y);
    thrust::reverse_iterator<Iterator> r_y = make_reverse_iterator(dptr_y+len);

    thrust::inclusive_scan(thrust::cuda::par.on(stream), r_x, r_x+len, r_y);
}

__global__
void adjust_ties_gpu(const numeric *x, const int *rank, numeric *y, uint32_t len)
{
    uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid < len)
    {
        y[tid] = x[rank[tid]];
    }

}

// adjust rank of x and save it to y
void adjust_ties(const numeric *x, const int *rank, numeric *y, uint32_t len , cudaStream_t stream)
{
    constexpr int num_thread = THREAD_PER_BLOCK;
    int num_block = (len + num_thread - 1)/num_thread;
    adjust_ties_gpu<<<num_block, num_thread, 0, stream>>>(x, rank, y, len);
}


__global__
void cwise_div_gpu(const numeric *x, const  numeric *y, numeric *z, uint32_t len)
{
    uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid < len)
    {
        z[tid] = x[tid]/y[tid];
    }

}


// Compute x./y and save the result to z
void cwise_div(const numeric *x, const numeric *y, numeric *z, uint32_t len, cudaStream_t stream)
{
    constexpr int num_thread = THREAD_PER_BLOCK;
    int num_block = (len + num_thread - 1)/num_thread;
    cwise_div_gpu<<<num_block, num_thread, 0, stream>>>(x, y, z, len);
}

void cumsum(numeric *x, uint32_t len, cudaStream_t stream)
{
    thrust::device_ptr<numeric> dev_ptr = thrust::device_pointer_cast(x);
    thrust::inclusive_scan(thrust::cuda::par.on(stream), dev_ptr, dev_ptr+len, dev_ptr);
}


__global__
void mult_add_gpu(numeric *z, const numeric *a, const numeric *b, const numeric *c, uint32_t len)
{
    uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid < len)
    {
        z[tid] = a[tid] * b[tid] - c[tid];
    }

}


// Set z = a*b - c
void mult_add(numeric *z, const numeric *a, const numeric *b, const numeric *c, uint32_t len,cudaStream_t stream)
{
    constexpr int num_thread = THREAD_PER_BLOCK;
    int num_block = (len + num_thread - 1)/num_thread;
    mult_add_gpu<<<num_block, num_thread, 0, stream>>>(z, a, b, c, len);
}


__global__
void coxval_gpu(const numeric *x, numeric *y, const numeric *z, numeric *val, uint32_t len)
{
    uint32_t i = blockIdx.x*(2*THREAD_PER_BLOCK) + threadIdx.x;
    if(i==0){
        val[0] = 0.0;
    }

    if(i < len){
        y[i] = (log(x[i]) - y[i]) * z[i];
    }

    if(i+THREAD_PER_BLOCK < len){
        y[i+THREAD_PER_BLOCK] = (log(x[i+THREAD_PER_BLOCK]) - y[i+THREAD_PER_BLOCK]) * z[i+THREAD_PER_BLOCK];
    }
    reduce_sum<THREAD_PER_BLOCK>(y, val, len);
}




// compute sum((log(x) - y) *z), x will be modified, result saved in val
void get_coxvalue(const numeric *x, numeric *y, const  numeric *z, numeric *val, uint32_t len, cudaStream_t stream)
{
    constexpr int num_thread = THREAD_PER_BLOCK;
    int num_block = (len + (2*num_thread) - 1)/(2*num_thread);
    coxval_gpu<<<num_block, num_thread, 0, stream>>>(x, y, z, val, len);
}

__global__
void update_parameters_gpu(numeric *B, const numeric *v, const numeric *g, const numeric *penalty_factor,
    uint32_t K, uint32_t p,numeric step_size, numeric lambda_1, numeric lambda_2)
{
    uint32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < p)
    {
        numeric ba;
        numeric lambdap1 = lambda_1*penalty_factor[i]*step_size;
        numeric lambdap2 = lambda_2*penalty_factor[i]*step_size;
        numeric norm = 0.0;
        for (uint32_t k = 0; k <K; ++k)
        {
            uint32_t ind = i+k*p;
            // gradient  descent
            B[ind] = v[ind] - step_size*g[ind];
            //soft-thresholding
            ba = fabs(B[ind]);
            B[ind] = signbit(lambdap1-ba)*copysign(ba-lambdap1, B[ind]);

            norm += B[ind]*B[ind];
        }
        // Group soft-thresholding
        norm = fmax(sqrt(norm), lambdap2);
        for(uint32_t k = 0; k <K; ++k)
        {
            uint32_t ind = i+k*p;
            B[ind] *= ((norm - lambdap2)/norm);
        }

    }

}


void update_parameters(cox_param &dev_param,
    uint32_t K,
    uint32_t p,
    numeric step_size,
    numeric lambda_1,
    numeric lambda_2)
{
    constexpr int num_thread = THREAD_PER_BLOCK;
    int num_block = (p + num_thread - 1)/num_thread;
    update_parameters_gpu<<<num_block, num_thread>>>(dev_param.B, dev_param.v, dev_param.grad, dev_param.penalty_factor,
                                                     K, p,step_size, lambda_1,lambda_2);

}


__global__
void ls_stop_v1_gpu(const numeric *B, const numeric *v, const numeric *g, numeric *result, uint32_t K, uint32_t p, numeric step_size)
{
    uint32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    uint32_t tid  = threadIdx.x;
    if(i==0){
        result[0] = 0.0;
    }

    numeric local = 0.0;
    if(i < K*p)
    {
        numeric diff = B[i] - v[i];
        local = g[i]*diff + diff*diff/(2*step_size);
    }
    __shared__ numeric sdata[THREAD_PER_BLOCK];
    sdata[tid] = local;

    __syncthreads();
    // This will only work when thread_per_block = 128
    // Perhaps  a better idea is to use a template Kernel?
    if (tid < 64) { sdata[tid] += sdata[tid + 64]; } 
    __syncthreads();

    if (tid < 32) warpSum<THREAD_PER_BLOCK>(sdata, tid);
    if(tid == 0){
        atomicAdd(result,sdata[0]);
    }
}



numeric ls_stop_v1(cox_param &dev_param, numeric step_size, uint32_t K, uint32_t p)
{
    constexpr int num_thread = THREAD_PER_BLOCK;
    int num_block = (K*p + num_thread - 1)/num_thread;
    ls_stop_v1_gpu<<<num_block, num_thread>>>(dev_param.B, dev_param.v, dev_param.grad, dev_param.ls_result, K, p, step_size);
    numeric result[1];
    cudaMemcpy(result, dev_param.ls_result, sizeof(numeric)*1, cudaMemcpyDeviceToHost);
    return result[0];
}


__global__
void ls_stop_v2_gpu(const numeric *B, const numeric *v, const numeric *g, const numeric *g_ls,
                    numeric *result, uint32_t K, uint32_t p, numeric step_size)
{
    uint32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i<2){
        result[i] = 0.0;
    }

    numeric local = 0.0;
    numeric diff = 0.0;
    if(i < K*p)
    {
        diff = B[i] - v[i];
        local = diff*diff;
    }
    __shared__ numeric sdata[256];
    sdata[threadIdx.x] = local;

    __syncthreads();
    // do reduction in shared mem
    for (int s=1; s < blockDim.x; s *=2)
    {
        int index = 2 * s * threadIdx.x;
        if (index < blockDim.x)
        {
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }
    // write result for this block to global mem
    if (threadIdx.x == 0){
        atomicAdd(result, sdata[0]);
    }
    // second term
    if(i < K*p)
    {
        local = diff*(g_ls[i]-g[i]);
    }

    sdata[threadIdx.x] = local;

    __syncthreads();
    // do reduction in shared mem
    for (int s=1; s < blockDim.x; s *=2)
    {
        int index = 2 * s * threadIdx.x;
        if (index < blockDim.x)
        {
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }
    // write result for this block to global mem
    if (threadIdx.x == 0){
        atomicAdd(result+1, sdata[0]);
    }
}





numeric ls_stop_v2(cox_param &dev_param, numeric step_size, uint32_t K, uint32_t p)
{
    constexpr int num_thread = 256;
    int num_block = (K*p + num_thread - 1)/num_thread;
    ls_stop_v2_gpu<<<num_block, num_thread>>>(dev_param.B, dev_param.v, dev_param.grad, dev_param.grad_ls,
                                                dev_param.ls_result, K, p, step_size);
    numeric result[2];
    cudaMemcpy(result, dev_param.ls_result, sizeof(numeric)*2, cudaMemcpyDeviceToHost);

    return (result[0]/(2*step_size) - abs(result[1]));
}


void nesterov_update(cox_param &dev_param, uint32_t K, uint32_t p, numeric weight_old, numeric weight_new, cudaStream_t stream, cublasHandle_t handle)
{
    numeric alpha =  (weight_old - 1)/weight_new + 1;
    numeric beta = (1 - weight_old)/weight_new;
    cublasSetStream(handle, stream);
    cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, p, K, &alpha, dev_param.B, p , &beta, dev_param.prev_B, p, dev_param.v, p);
}


__global__
void max_diff_gpu(numeric *A, numeric *B, numeric *result, uint32_t len)
{
    uint32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i == 0)
    {
        result[0] = 0.0;
    }
    numeric local = 0;
    if(i < len)
    {
        local = fabs(A[i] - B[i]);
    }

    __shared__ numeric sdata[256];
    sdata[threadIdx.x] = local;

    __syncthreads();
    // do reduction in shared mem
    for (int s=1; s < blockDim.x; s *=2)
    {
        int index = 2 * s * threadIdx.x;

        if (index < blockDim.x)
        {
            sdata[index] = fmax(sdata[index + s], sdata[index]);
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (threadIdx.x == 0){
        atomicMax(result, sdata[0]);
    }


}

numeric max_diff(cox_param &dev_param, uint32_t K, uint32_t p)
{
    constexpr int num_thread = 256;
    int num_block = (K*p + num_thread - 1)/num_thread;
    max_diff_gpu<<<num_block, num_thread>>>(dev_param.B, dev_param.prev_B, dev_param.change, K*p);
    numeric result[1];
    cudaMemcpy(result, dev_param.change, sizeof(numeric)*1, cudaMemcpyDeviceToHost);
    return result[0];
}

// Set B = A
void cublas_copy(numeric *A, numeric *B, uint32_t len, cudaStream_t stream, cublasHandle_t handle)
{
    cublasSetStream(handle, stream);
    cublasScopy(handle, len,A, 1,B, 1);
}


void permute_by_order(numeric *x, numeric *y, int *o, uint32_t len, cudaStream_t stream)
{
    thrust::device_ptr<numeric> dptr_x = thrust::device_pointer_cast<numeric>(x);
    thrust::device_ptr<numeric> dptr_y = thrust::device_pointer_cast<numeric>(y);
    thrust::device_ptr<int> dptr_o = thrust::device_pointer_cast<int>(o);
    PermIter iter(dptr_x, dptr_o);
    thrust::copy_n(thrust::cuda::par.on(stream), iter, len, y);

}