#ifndef GPU_FUNC_H_
#define GPU_FUNC_H_

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "cublas_v2.h"
#include <thrust/execution_policy.h>
#include "proxgpu_types.h"


typedef struct cox_data{
    numeric *X;
    numeric *status;
    int *rankmin;
    int *rankmax;
    int *order;
    int *rev_order;
} cox_data;

typedef struct cox_cache{
    numeric *outer_accumu;
    numeric *eta;
    numeric *exp_eta;
    numeric *exp_accumu;
    numeric *residual;
    numeric *B_col_norm;
    numeric *cox_val;
    numeric *order_cache;

} cox_cache;

typedef struct cox_param{
    numeric *B;
    numeric *v;
    numeric *grad;
    numeric *prev_B;
    numeric *penalty_factor;
    numeric *ls_result;
    numeric *grad_ls;
    numeric *change;

} cox_param;

void allocate_device_memory(cox_data &dev_data, cox_cache &dev_cache, cox_param &dev_param, uint32_t total_cases, uint32_t K, uint32_t p);

void free_device_memory(cox_data &dev_data, cox_cache &dev_cache, cox_param &dev_param);

// Compute trans(A)*B and save the result to C, trans(A) is N by p, B is p by K
void compute_product(numeric *A, numeric *B, numeric *C, 
                     uint32_t N, uint32_t p, uint32_t K, cudaStream_t stream, cublasHandle_t handle, cublasOperation_t trans);

void apply_exp(const numeric *x, numeric *ex, uint32_t len, cudaStream_t stream);
void rev_cumsum(numeric *x, numeric *y, uint32_t len, cudaStream_t stream);
void adjust_ties(const numeric *x, const int *rank, numeric *y, uint32_t len , cudaStream_t stream);
void cwise_div(const numeric *x, const numeric *y, numeric *z, uint32_t len, cudaStream_t stream);
void cumsum(numeric *x, uint32_t len, cudaStream_t stream);

void mult_add(numeric *z, const numeric *a, const numeric *b, const numeric *c, uint32_t len,cudaStream_t stream);

void get_coxvalue(const numeric *x, numeric *y, const numeric *z, numeric *val, uint32_t len, cudaStream_t stream);

void update_parameters(cox_param &dev_param,
                       uint32_t K,
                       uint32_t p,
                       numeric step_size,
                       numeric lambda_1,
                       numeric lambda_2);

numeric ls_stop_v1(cox_param &dev_param, numeric step_size, uint32_t K, uint32_t p);

numeric ls_stop_v2(cox_param &dev_param, numeric step_size, uint32_t K, uint32_t p);

void nesterov_update(cox_param &dev_param, uint32_t K, uint32_t p, numeric weight_old, numeric weight_new, cudaStream_t stream, cublasHandle_t handle);

numeric max_diff(cox_param &dev_param, uint32_t K, uint32_t p);

void cublas_copy(numeric *A, numeric *B, uint32_t len, cudaStream_t stream, cublasHandle_t handle);

// Set y = x[o];
void permute_by_order(numeric *x, numeric *y, int *o, uint32_t len, cudaStream_t stream);
#endif