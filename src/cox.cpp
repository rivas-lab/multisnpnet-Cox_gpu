#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include "gpu_func.h"
#include <sys/time.h>
#include "proxgpu_types.h"


// Compute the gradient at B
numeric get_value_only(cox_cache &dev_cache,
                  cox_data &dev_data,
                  cox_param &dev_param,
                  numeric *B, // B is a device pointer
                  MatrixXd & host_cache, // Used to store eta, residual to reorder
                  int p,
                  cublasHandle_t handle,
                  cudaStream_t *copy_stream,
                  numeric *cox_val_host)
{
    int N = host_cache.rows();
    int K = host_cache.cols();
    compute_product(dev_data.X, B, dev_cache.order_cache, N, p, K, 0, handle, CUBLAS_OP_N);
    for(int k = 0; k< K; ++k){
        permute_by_order(dev_cache.order_cache + k*N, dev_cache.eta+k*N, dev_data.order+k*N, N, copy_stream[k]);
    }
    cudaDeviceSynchronize();
    apply_exp(dev_cache.eta, dev_cache.exp_eta, N*K, 0);
    for(int k = 0; k < K; ++k){
        // Save rev_cumsum result to dev_cache.outer_accumu to avoid more cache variables
        rev_cumsum(dev_cache.exp_eta+k*N, dev_cache.outer_accumu+k*N ,N, copy_stream[k]);
        adjust_ties(dev_cache.outer_accumu+k*N, dev_data.rankmin+k*N, dev_cache.exp_accumu+k*N, N, copy_stream[k]);
        
        get_coxvalue(dev_cache.exp_accumu+k*N, dev_cache.eta+k*N, dev_data.status+k*N, dev_cache.cox_val+k, N, copy_stream[k]);
        cudaMemcpyAsync(cox_val_host+k, dev_cache.cox_val+k, sizeof(numeric)*1, cudaMemcpyDeviceToHost, copy_stream[k]);
    }
    numeric val =  0.0;
    cudaDeviceSynchronize();
    for(int k = 0; k < K; ++k)
    {
        val += cox_val_host[k];
    }
    return val;
}



// Compute the gradient at B and save the result to dev_grad
numeric get_gradient(cox_cache &dev_cache,
                  cox_data &dev_data,
                  cox_param &dev_param,
                  numeric *dev_grad,
                  numeric *B, // B is a device pointer
                  MatrixXd & host_cache, // Used to store eta, residual to reorder
                  int p,
                  cublasHandle_t handle,
                  cudaStream_t *copy_stream,
                  bool get_val = false,
                  numeric *cox_val_host=0)
{
    int N = host_cache.rows();
    int K = host_cache.cols();
    compute_product(dev_data.X, B, dev_cache.order_cache, N, p, K, 0, handle, CUBLAS_OP_N);
    for(int k = 0; k< K; ++k){
        permute_by_order(dev_cache.order_cache + k*N, dev_cache.eta+k*N, dev_data.order+k*N, N, copy_stream[k]);
    }
    cudaDeviceSynchronize();
    apply_exp(dev_cache.eta, dev_cache.exp_eta, N*K, 0);
    for(int k = 0; k < K; ++k){
        // Save rev_cumsum result to dev_cache.outer_accumu to avoid more cache variables
        rev_cumsum(dev_cache.exp_eta+k*N, dev_cache.outer_accumu+k*N ,N, copy_stream[k]);
        adjust_ties(dev_cache.outer_accumu+k*N, dev_data.rankmin+k*N, dev_cache.exp_accumu+k*N, N, copy_stream[k]);
    }
    cudaDeviceSynchronize();
    // Above is  _update_exp()
    // Below is _update_outer()
    // Save the result of division to residual to avoid more cache variables
    cwise_div(dev_data.status, dev_cache.exp_accumu, dev_cache.residual,  N*K, 0);

    for(int k = 0; k < K; ++k){
        cumsum(dev_cache.residual+k*N, N, copy_stream[k]);
        adjust_ties(dev_cache.residual+k*N, dev_data.rankmax+k*N, dev_cache.outer_accumu+k*N, N, copy_stream[k]);
    }
    cudaDeviceSynchronize();
    mult_add(dev_cache.order_cache, dev_cache.exp_eta, dev_cache.outer_accumu, dev_data.status, N*K, 0);
    // residuals almost ready, need to reorder them:
    for(int k = 0; k< K; ++k){
        permute_by_order(dev_cache.order_cache + k*N, dev_cache.residual+k*N, dev_data.rev_order+k*N, N, copy_stream[k]);
    }
    // reiduals ready, and stored in host_cache, now we can get gradient
    compute_product(dev_data.X, dev_cache.residual, dev_grad, p, N, K, 0,handle, CUBLAS_OP_T);
    numeric val = 0.0;
    if(get_val){
        for(int k = 0; k < K; ++k){
            get_coxvalue(dev_cache.exp_accumu+k*N, dev_cache.eta+k*N, dev_data.status+k*N, dev_cache.cox_val+k, N, copy_stream[k]);
            cudaMemcpyAsync(cox_val_host+k, dev_cache.cox_val+k, sizeof(numeric)*1, cudaMemcpyDeviceToHost, copy_stream[k]);
        }
        cudaDeviceSynchronize();
        for(int k = 0; k < K; ++k)
        {
            val += cox_val_host[k];
        }
    }
    return val;
}


// [[Rcpp::export]]
Rcpp::List solve_path(Rcpp::NumericMatrix X,
                       Rcpp::NumericMatrix status,
                       Rcpp::IntegerMatrix rankmin,
                       Rcpp::IntegerMatrix rankmax,
                       Rcpp::IntegerMatrix order_mat,
                       Rcpp::IntegerMatrix rev_order_mat,
                       Rcpp::NumericMatrix B0,
                       Rcpp::NumericVector lambda_1_all,
                       Rcpp::NumericVector lambda_2_all,
                       VectorXd pfac,
                       double step_size = 1.0,
                       int niter=2000,
                       double linesearch_beta = 1.1,
                       double eps=1e-5 // convergence criteria
                       )
{
    // B is a long and skinny matrix now! rows are features and cols are responses
    const int p = B0.rows();
    const int K = B0.cols();
    const int N = X.rows();
    // Create CUDA streams and handle
    cublasHandle_t handle;
    cublasCreate(&handle);
    cudaStream_t nest_stream;
    cudaStreamCreate(&nest_stream); // Stream for Nesterov acceleration
    
    // Use these streams to copy data to device
    cudaStream_t *copy_stream = (cudaStream_t *) malloc(K * sizeof(cudaStream_t));
    for (int k = 0; k<K; ++k)
    {
        cudaStreamCreate(&copy_stream[k]);
    }


    cox_data dev_data;
    cox_cache dev_cache;
    cox_param dev_param;

    // Allocate host variables
    numeric *cox_val_host=(numeric *)malloc(sizeof(numeric)*K);
    MatrixXd host_B(p,K);
    MatrixXd host_cache(N, K); // This will hold the vectors that needs to be reordered and the residuals



    allocate_device_memory(dev_data, dev_cache, dev_param, N, K, p);

    // initialize parameters on the device
    cudaMemcpy(dev_param.B, &B0(0,0), sizeof(numeric)*K*p, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_param.v, &B0(0,0), sizeof(numeric)*K*p, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_param.penalty_factor, &pfac(0), sizeof(numeric)*p, cudaMemcpyHostToDevice);

    // Copy the data
    cudaMemcpy(dev_data.X, &X(0,0), sizeof(numeric)*N*p, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_data.status, &status(0,0), sizeof(numeric)*N*K, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_data.rankmin, &rankmin(0,0), sizeof(int)*N*K, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_data.rankmax, &rankmax(0,0), sizeof(int)*N*K, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_data.order, &order_mat(0,0), sizeof(int)*N*K, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_data.rev_order, &rev_order_mat(0,0), sizeof(int)*N*K, cudaMemcpyHostToDevice);


    cudaDeviceSynchronize();

    // check for error
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        // print the CUDA error message and exit
        std::cout << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        Rcpp::stop("CUDA error occured, most likely a memory allocation problem.\n"); 
    }


    numeric cox_val;
    numeric cox_val_next;
    numeric rhs_ls; // right-hand side of line search condition
    numeric diff; // Max norm of the difference of two consecutive iterates
    numeric lambda_1;
    numeric lambda_2;
    const int num_lambda = lambda_1_all.size();
    numeric step_size_intial = step_size;
    Rcpp::List result(num_lambda);
    bool stop; // Stop line searching
    numeric value_change;
    numeric weight_old, weight_new;
    // Initialization done, starting solving the path
    struct timeval start, end;
    for (int lam_ind = 0; lam_ind < num_lambda; ++lam_ind){
        gettimeofday(&start, NULL);

        lambda_1 = lambda_1_all[lam_ind];
        lambda_2 = lambda_2_all[lam_ind];
        weight_old = 1.0;
        step_size = step_size_intial;
        cublas_copy(dev_param.B, dev_param.v, K*p, 0, handle);
        // Inner iteration
        for (int i = 0; i < niter; ++i)
        {

            // Set prev_B = B
            cublas_copy(dev_param.B, dev_param.prev_B, K*p, copy_stream[0], handle);
            // Wait for Nesterov weight update
            cudaStreamSynchronize(nest_stream);
            // Update the gradient at v, compute cox_val at v
            cox_val = get_gradient(dev_cache,
                                    dev_data,
                                    dev_param,
                                    dev_param.grad,
                                    dev_param.v, // B is a device pointer
                                    host_cache, // Used to store eta, residual to reorder
                                    p,
                                    handle,
                                    copy_stream,
                                    true,
                                    cox_val_host);
            // Enter line search
            while(true)
            {
                // Update  B
                update_parameters(dev_param,
                                    K,
                                    p,
                                    step_size,
                                    lambda_1,
                                    lambda_2);

                // Get cox_val at updated B
                cox_val_next = get_value_only(dev_cache,
                                                dev_data,
                                                dev_param,
                                                dev_param.B, // B is a device pointer
                                                host_cache, // Used to store eta, residual to reorder
                                                p,
                                                handle,
                                                copy_stream,
                                                cox_val_host);


                // This block are the line search conditions
                if(!std::isfinite(cox_val_next)){
                    stop = false;
                } else {
                    rhs_ls = cox_val + ls_stop_v1(dev_param, step_size,K,p);
                    stop = (cox_val_next <= rhs_ls);
                }

                if (stop)
                {
                    value_change = abs(cox_val_next - cox_val)/fmax(1.0, abs(cox_val));
                    break;
                }
                step_size /= linesearch_beta;
            }

            diff = max_diff(dev_param, K, p);
            if (diff < eps)
            {
                std::cout << "convergence based on parameter change reached in " << i <<" iterations\n";
                std::cout << "current step size is " << step_size << std::endl;
                gettimeofday(&end, NULL);
                double delta  = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
                std::cout <<  "elapsed time is " << delta << " seconds" << std::endl;
                Rcpp::checkUserInterrupt();
                break;
            }

            if(value_change < 1e-10){
                std::cout << "convergence based on value change reached in " << i <<" iterations\n";
                std::cout << "current step size is " << step_size << std::endl;
                gettimeofday(&end, NULL);
                double delta  = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
                std::cout <<  "elapsed time is " << delta << " seconds" << std::endl;
                Rcpp::checkUserInterrupt();
                break;
            }

             // Nesterov weight
            weight_new = 0.5*(1+sqrt(1+4*weight_old*weight_old));
            nesterov_update(dev_param,K,p, weight_old, weight_new, nest_stream, handle);
            weight_old = weight_new;

            if (i != 0 && i % 100 == 0)
            {
                std::cout << "reached " << i << " iterations\n";
                gettimeofday(&end, NULL);
                double delta  = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
                std::cout <<  "elapsed time is " << delta  << " seconds" << std::endl;
                Rcpp::checkUserInterrupt();
            }

        }
        cudaMemcpy(&host_B(0,0), dev_param.B, sizeof(numeric)*K*p, cudaMemcpyDeviceToHost);
        result[lam_ind] = host_B;
        std::cout << "Solution for the " <<  lam_ind+1 << "th lambda pair is obtained\n";
    }




    free_device_memory(dev_data, dev_cache, dev_param);
    cublasDestroy(handle);
    for (int si = 0; si< K;++si){
        cudaStreamDestroy(copy_stream[si]);
    }
    cudaStreamDestroy(nest_stream);
    free(copy_stream);
    free(cox_val_host);
    return result;
}