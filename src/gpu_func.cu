#include "gpu_func.h"
#include <stdlib.h>
#include <cub/device/device_scan.cuh>


#define THREAD_PER_BLOCK 128


cudaError_t allocate_device_memory(cox_data &dev_data, cox_cache &dev_cache, cox_param &dev_param, 
                                    uint32_t total_cases, uint32_t K, uint32_t p)
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
    cudaMalloc((void**)&dev_param.penalty_factor, sizeof(numeric) * p);
    cudaMalloc((void**)&dev_param.ls_result, sizeof(numeric) * 2);

    cudaError_t error = cudaGetLastError();
    return error;

}


void free_device_memory(cox_data &dev_data, cox_cache &dev_cache, cox_param &dev_param)
{
    cudaFree(dev_data.X);
    cudaFree(dev_data.status);
    cudaFree(dev_data.rankmax);
    cudaFree(dev_data.rankmin);
    cudaFree(dev_data.order);
    cudaFree(dev_data.rev_order);

    cudaFree(dev_cache.outer_accumu);
    cudaFree(dev_cache.eta);
    cudaFree(dev_cache.exp_eta);
    cudaFree(dev_cache.exp_accumu);
    cudaFree(dev_cache.residual);
    cudaFree(dev_cache.B_col_norm);
    cudaFree(dev_cache.cox_val);
    cudaFree(dev_cache.order_cache);

    cudaFree(dev_param.B);
    cudaFree(dev_param.v);
    cudaFree(dev_param.grad);
    cudaFree(dev_param.prev_B);
    cudaFree(dev_param.penalty_factor);
    cudaFree(dev_param.ls_result);
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
    uint32_t N, uint32_t p, uint32_t K, cublasHandle_t handle, cublasOperation_t trans)
{
    numeric alpha = 1.0;
    numeric beta = 0.0;
    int lda = (trans == CUBLAS_OP_N)?N:p;
    cublasSgemm(handle, trans, CUBLAS_OP_N, 
        N, K, p, &alpha, 
        A, lda, B, p, &beta, C, N);

    cudaDeviceSynchronize();
}


__global__ void apply_exp_gpu(const numeric *x, numeric *ex, uint32_t len)
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

__global__ void reverse_inplace(numeric *x, uint32_t len)
{
    uint32_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < (len/2))
    {
        numeric tmp = x[len-i-1];
        x[len-i-1] = x[i];
        x[i] = tmp;
    }
}

__global__ void reverse(const numeric *x, numeric *y, uint32_t len)
{
    uint32_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < len)
    {
        y[i] = x[len-i-1];
    }
}

// do rev_cumsum of x and save it to y
void rev_cumsum(numeric *x, numeric *y, uint32_t len, 
                void *d_temp_storage,
                size_t & temp_storage_bytes,
                cudaStream_t stream)
{
    constexpr int num_thread = THREAD_PER_BLOCK;
    int block_half = ((len/2) + num_thread - 1)/num_thread;
    int block = (len + num_thread - 1)/num_thread;
    reverse<<<block, num_thread, 0, stream>>>(x, y, len);
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, y, y, len, stream);
    reverse_inplace<<<block_half, num_thread, 0, stream>>>(y, len);

}


__global__ void adjust_ties_gpu(const numeric *x, const int *rank, numeric *y, uint32_t len)
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



__global__ void cwise_div_gpu(const numeric *x, const  numeric *y, numeric *z, uint32_t len)
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


void cumsum(numeric *x, uint32_t len, void *d_temp_storage, size_t & temp_storage_bytes, cudaStream_t stream)
{
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, x, x, len, stream);
}



__global__ void mult_add_gpu(numeric *z, const numeric *a, const numeric *b, const numeric *c, uint32_t len)
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


__global__ void coxval_gpu(const numeric *x, numeric *y, const numeric *z, numeric *val, uint32_t len)
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


__global__ void update_parameters_gpu(numeric *B, const numeric *v, const numeric *g, const numeric *penalty_factor,
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

__global__ void ls_stop_v1_gpu(const numeric *B, const numeric *v, const numeric *g, numeric *result, uint32_t K, uint32_t p, numeric step_size)
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



void nesterov_update(cox_param &dev_param, uint32_t K, uint32_t p, numeric weight_old, numeric weight_new, cublasHandle_t handle)
{
    numeric alpha =  (weight_old - 1)/weight_new + 1;
    numeric beta = (1 - weight_old)/weight_new;
    cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, p, K, &alpha, dev_param.B, p , &beta, dev_param.prev_B, p, dev_param.v, p);

    cudaDeviceSynchronize();
}


// Set B = A
void cublas_copy(numeric *A, numeric *B, uint32_t len, cublasHandle_t handle)
{
    cublasScopy(handle, len,A, 1,B, 1);

    cudaDeviceSynchronize();
}

// Could there be a better way to do this?
__global__ void permute_by_order_gpu(const numeric *x, numeric *y, const int *o, uint32_t len)
{
    uint32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < len){
        y[i] = x[o[i]];
    }
}

void permute_by_order(numeric *x, numeric *y, int *o, uint32_t len, cudaStream_t stream)
{
    // thrust::device_ptr<numeric> dptr_x = thrust::device_pointer_cast<numeric>(x);
    // thrust::device_ptr<numeric> dptr_y = thrust::device_pointer_cast<numeric>(y);
    // thrust::device_ptr<int> dptr_o = thrust::device_pointer_cast<int>(o);
    // PermIter iter(dptr_x, dptr_o);
    // thrust::copy_n(thrust::cuda::par.on(stream), iter, len, y);
    constexpr int num_thread = THREAD_PER_BLOCK;
    int num_block = (len + num_thread - 1)/num_thread;
    permute_by_order_gpu<<<num_block, num_thread, 0, stream>>>(x, y, o, len);

}


void get_cub_scan_info(void *d_temp_storage, size_t &temp_storage_bytes, numeric *d_in, numeric *d_out, int num_items)
{
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
}
