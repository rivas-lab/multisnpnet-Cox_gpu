#include <iostream>
#include "gpu_func.h"
#include <sys/time.h>
#include <vector>
#include <RcppEigen.h>

class Cox_GPU
{
    uint32_t N; // Number of observations
    uint32_t K; // Number of responses, after  removing some  responses
    uint32_t p; // Number of variables
    uint32_t K0; // Number of responses at the beginning

    cox_data dev_data;
    cox_cache dev_cache;
    numeric *cox_val_host;
    size_t temp_storage_bytes;
    std::vector<void *> d_temp_storage;

    // Here B has to be a device pointer
    void update_exp(numeric *B,
                    std::vector<cudaStream_t> &streams,
                    cublasHandle_t handle)
    {
        int num_stream = streams.size();
        compute_product(dev_data.X, B, dev_cache.order_cache, N, p, K, handle, CUBLAS_OP_N);

        for (uint32_t k = 0; k < K; ++k)
        {
            permute_by_order(dev_cache.order_cache + k * N, dev_cache.eta + k * N, dev_data.order + k * N, N, streams[k % num_stream]);
        }
        cudaDeviceSynchronize();

        apply_exp(dev_cache.eta, dev_cache.exp_eta, N * K, 0);
        for (uint32_t k = 0; k < K; ++k)
        {
            // Save rev_cumsum result to dev_cache.outer_accumu to avoid more cache variables
            rev_cumsum(dev_cache.exp_eta + k * N, dev_cache.outer_accumu + k * N, N, d_temp_storage[k], temp_storage_bytes, streams[k % num_stream]);
            adjust_ties(dev_cache.outer_accumu + k * N, dev_data.rankmin + k * N, dev_cache.exp_accumu + k * N, N, streams[k % num_stream]);
        }
    }

    void compute_residual(numeric *B,
                          std::vector<cudaStream_t> &streams,
                          cublasHandle_t handle)
    {
        int num_stream = streams.size();
        update_exp(B, streams, handle);
        cudaDeviceSynchronize();
        cwise_div(dev_data.status, dev_cache.exp_accumu, dev_cache.residual, N * K, 0);
        for (uint32_t k = 0; k < K; ++k)
        {
            cumsum(dev_cache.residual + k * N, N, d_temp_storage[k], temp_storage_bytes, streams[k % num_stream]);
            adjust_ties(dev_cache.residual + k * N, dev_data.rankmax + k * N, dev_cache.outer_accumu + k * N, N, streams[k % num_stream]);
        }
        cudaDeviceSynchronize();
        mult_add(dev_cache.order_cache, dev_cache.exp_eta, dev_cache.outer_accumu, dev_data.status, N * K, 0);
        // residuals almost ready, need to reorder them:
        for (uint32_t k = 0; k < K; ++k)
        {
            permute_by_order(dev_cache.order_cache + k * N, dev_cache.residual + k * N, dev_data.rev_order + k * N, N, streams[k % num_stream]);
        }
        cudaDeviceSynchronize();
    }

public:
    // host variables
    Eigen::Matrix<numeric, Eigen::Dynamic, Eigen::Dynamic> host_B;
    Eigen::Matrix<numeric, Eigen::Dynamic, Eigen::Dynamic> host_residual;
    cox_param dev_param; // make the parameters public

    Cox_GPU(uint32_t N,
            uint32_t K,
            uint32_t p,
            uint32_t K0,
            double *X,
            double *status,
            double *B0,
            double *pfac,
            int *rankmin,
            int *rankmax,
            int *order,
            int *rev_order) : N(N), K(K), p(p), K0(K0)
    {
        std::cout << "Number of observations: " << N << std::endl;
        std::cout << "Number of responses: " << K << std::endl;
        std::cout << "Number of variables: " << p << std::endl;

        size_t totalm, freem;
        cudaMemGetInfo(&freem, &totalm);
        std::cout << "Total memory available: " << ((double)totalm) / 1e9 << " GB.\t";
        std::cout << ((double)freem) / 1e9 << " GB free before allocation\n";

        cudaError_t error = allocate_device_memory(dev_data, dev_cache, dev_param, N, K, p, K0);
        if (error != cudaSuccess)
        {
            // print the CUDA error message and exit
            std::cout << "CUDA error: " << cudaGetErrorString(error) << std::endl;
            Rcpp::stop("CUDA malloc failed, not enough memory\n");
        }
        std::cout << "GPU memory allocation successful.\t";
        cudaMemGetInfo(&freem, &totalm);
        std::cout << ((double)freem) / 1e9 << " GB free after allocation\n";

        cox_val_host = (numeric *)malloc(K * sizeof(numeric));
        host_B.resize(p, K0);
        host_residual.resize(N, K);

        struct timeval start, end;
        std::cout << "Casting data to float32...\t";
        gettimeofday(&start, NULL);
        // Do everything in single precision.
        Eigen::Map<Eigen::MatrixXd> Xmap(X, N, p);
        Eigen::Map<Eigen::MatrixXd> statusmap(status, N, K);
        Eigen::Map<Eigen::MatrixXd> Bmap(B0, p, K0);
        Eigen::Map<Eigen::VectorXd> pfacmap(pfac, p);
        Eigen::MatrixXf Xf(Xmap.cast<float>());
        Eigen::MatrixXf statusf(statusmap.cast<float>());
        Eigen::MatrixXf Bf(Bmap.cast<float>());
        Eigen::VectorXf pfacf(pfacmap.cast<float>());
        gettimeofday(&end, NULL);
        double delta = ((end.tv_sec - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
        std::cout << "Succeed! Elapsed time: "<< delta << " seconds.\n";

        std::cout << "Copying data to the GPU...\t";

        // initialize parameters on the device
        cudaMemcpy(dev_param.B, &Bf(0, 0), sizeof(numeric) * K0 * p, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_param.v, &Bf(0, 0), sizeof(numeric) * K * p, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_param.penalty_factor, &pfacf(0), sizeof(numeric) * p, cudaMemcpyHostToDevice);

        // Copy the data
        cudaMemcpy(dev_data.X, &Xf(0, 0), sizeof(numeric) * N * p, cudaMemcpyHostToDevice);


        cudaMemcpy(dev_data.status, &statusf(0, 0), sizeof(numeric) * N * K, cudaMemcpyHostToDevice);

        cudaMemcpy(dev_data.rankmin, rankmin, sizeof(int) * N * K, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_data.rankmax, rankmax, sizeof(int) * N * K, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_data.order, order, sizeof(int) * N * K, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_data.rev_order, rev_order, sizeof(int) * N * K, cudaMemcpyHostToDevice);
        std::cout << "Succeed!\n";

        temp_storage_bytes = 0;
        for (uint32_t k = 0; k < K; ++k)
        {
            d_temp_storage.push_back(NULL);
            get_cub_scan_info(d_temp_storage[k], temp_storage_bytes, dev_data.status, dev_data.status, N);
            cudaMalloc(&d_temp_storage[k], temp_storage_bytes);
        }
        std::cout << "temp storage CUB scan is " << temp_storage_bytes << "bytes \n";
        std::cout << "Checking CUDA errors...\t";
        cudaError_t error2 = cudaGetLastError();
        if (error2 != cudaSuccess)
        {
            // print the CUDA error message and exit
            std::cout << "CUDA error: " << cudaGetErrorString(error2) << std::endl;
            Rcpp::stop("A CUDA error occurred during cudaMemcpy or CUB initialization. Stop!\n");
        }
        std::cout << "None. Cox GPU object constructed successfully!\n";
    }

    ~Cox_GPU()
    {
        free(cox_val_host);
        free_device_memory(dev_data, dev_cache, dev_param);
        for (uint32_t k = 0; k < K; ++k)
        {
            cudaFree(d_temp_storage[k]);
        }
    }

    numeric get_value_only(numeric *B,
                           std::vector<cudaStream_t> &streams,
                           cublasHandle_t handle)
    {
        int num_stream = streams.size();
        update_exp(B, streams, handle);
        for (uint32_t k = 0; k < K; ++k)
        {
            get_coxvalue(dev_cache.exp_accumu + k * N, dev_cache.eta + k * N, dev_data.status + k * N, dev_cache.cox_val + k,
                         N, streams[k % num_stream]);
            cudaMemcpyAsync(cox_val_host + k, dev_cache.cox_val + k, sizeof(numeric) * 1, cudaMemcpyDeviceToHost, streams[k % num_stream]);
        }
        numeric val = 0.0;
        cudaDeviceSynchronize();
        for (uint32_t k = 0; k < K; ++k)
        {
            val += cox_val_host[k];
        }
        return val;
    }

    numeric get_gradient(numeric *B,
                         std::vector<cudaStream_t> &streams,
                         cublasHandle_t handle,
                         bool get_val = false)
    {
        compute_residual(B, streams, handle);
        compute_product(dev_data.X, dev_cache.residual, dev_param.grad, p, N, K, handle, CUBLAS_OP_T);

        numeric val = 0.0;
        if (get_val)
        {
            int num_stream = streams.size();
            for (uint32_t k = 0; k < K; ++k)
            {
                get_coxvalue(dev_cache.exp_accumu + k * N, dev_cache.eta + k * N, dev_data.status + k * N, dev_cache.cox_val + k, N, streams[k % num_stream]);
                cudaMemcpyAsync(cox_val_host + k, dev_cache.cox_val + k, sizeof(numeric) * 1, cudaMemcpyDeviceToHost, streams[k % num_stream]);
            }
            cudaDeviceSynchronize();
            for (uint32_t k = 0; k < K; ++k)
            {
                val += cox_val_host[k];
            }
        }
        return val;
    }

    void get_param(std::vector<cudaStream_t> &streams, cublasHandle_t handle)
    {
        compute_residual(dev_param.B, streams, handle);
        cudaMemcpy(&host_B(0, 0), dev_param.B, sizeof(numeric) * p * K0, cudaMemcpyDeviceToHost);
        cudaMemcpy(&host_residual(0, 0), dev_cache.residual, sizeof(numeric) * N * K, cudaMemcpyDeviceToHost);
    }

    void step(numeric step_size, numeric lambda_1, numeric lambda_2)
    {
        update_parameters(dev_param, K, p, K0, step_size, lambda_1, lambda_2);
    }

    bool ls_stop(numeric cox_val, numeric cox_val_next, numeric step_size)
    {
        numeric rhs = cox_val + ls_stop_v1(dev_param, step_size, K, p);
        return (cox_val_next <= rhs);
    }
};

// [[Rcpp::export]]
Rcpp::List solve_path(Rcpp::NumericMatrix X,
                      Rcpp::NumericMatrix status,
                      Rcpp::IntegerMatrix rankmin,
                      Rcpp::IntegerMatrix rankmax,
                      Rcpp::IntegerMatrix order,
                      Rcpp::IntegerMatrix rev_order,
                      Rcpp::NumericMatrix B0,
                      Rcpp::NumericVector lambda_1_all,
                      Rcpp::NumericVector lambda_2_all,
                      Rcpp::NumericVector pfac,
                      double step_sized = 1.0,
                      int niter = 2000,
                      double linesearch_betad = 1.1
)
{
    // B is a long and skinny matrix now! rows are features and cols are responses
    // B0 has  K0 columns, which is larger than K,
    // The extra columns are used only in proximal steps
    uint32_t p = B0.rows();
    uint32_t K0 = B0.cols();
    uint32_t N = X.rows();
    uint32_t K = status.cols();

    Cox_GPU prob(N, K, p, K0, &X(0, 0), &status(0, 0), &B0(0, 0), &pfac[0],
                 &rankmin(0, 0), &rankmax(0, 0), &order(0, 0), &rev_order(0, 0));

    float linesearch_beta = (float)linesearch_betad;

    // Create CUDA streams and handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Use these streams to copy data to device
    std::vector<cudaStream_t> streams(K);
    for (uint32_t k = 0; k < K; ++k)
    {
        cudaStreamCreate(&streams[k]);
    }

    const int num_lambda = lambda_1_all.size();
    float step_size_intial = (float)step_sized;
    bool stop = false;
    float value_change;
    float cox_val;
    float cox_val_next;
    Rcpp::List result(num_lambda);
    Rcpp::List residual_result(num_lambda);
    // Initialization done, starting solving the path
    struct timeval start, end;
    bool nan_stop = false;
    for (int lam_ind = 0; lam_ind < num_lambda; ++lam_ind)
    {
        gettimeofday(&start, NULL);

        float lambda_1 = (float)lambda_1_all[lam_ind];
        float lambda_2 = (float)lambda_2_all[lam_ind];
        float weight_old = 1.0;
        float step_size = step_size_intial;
        cublas_copy(prob.dev_param.B, prob.dev_param.v, K * p, handle);
        // Inner iteration
        for (int i = 0; i < niter; ++i)
        {
            // Set prev_B = B
            cublas_copy(prob.dev_param.B, prob.dev_param.prev_B, K * p, handle);

            // Update the gradient at v, compute cox_val at v
            cox_val = prob.get_gradient(prob.dev_param.v, streams, handle, true);

            // Enter line search
            while (true)
            {
                // Update  B
                prob.step(step_size, lambda_1, lambda_2);

                // Get cox_val at updated B
                cox_val_next = prob.get_value_only(prob.dev_param.B, streams, handle);

                // This block are the line search conditions
                if (!std::isfinite(cox_val_next))
                {
                    stop = (step_size < 1e-6);
                    if(stop){
                        nan_stop = true;
                        goto terminate_nan;
                    }
                }
                else
                {
                    stop = prob.ls_stop(cox_val, cox_val_next, step_size);
                }


                if (stop)
                {
                    value_change = abs(cox_val_next - cox_val) / fmax(1.0, abs(cox_val));
                    break;
                }
                step_size /= linesearch_beta;
            }

            if (value_change < 5e-7)
            {
                std::cout << "convergence based on value change reached in " << i << " iterations\n";
                // std::cout << "current step size is " << step_size << std::endl;
                gettimeofday(&end, NULL);
                double delta = ((end.tv_sec - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
                std::cout << "elapsed time is " << delta << " seconds" << std::endl;

                // std::cout << "Objective value (smooth) at convergence is " << cox_val_next << std::endl;
                Rcpp::checkUserInterrupt();
                break;
            }

            // Nesterov weight
            float weight_new = 0.5 * (1 + sqrt(1 + 4 * weight_old * weight_old));
            nesterov_update(prob.dev_param, K, p, weight_old, weight_new, handle);
            weight_old = weight_new;

            if (i != 0 && i % 100 == 0)
            {
                std::cout << "reached " << i << " iterations\n";
                gettimeofday(&end, NULL);
                double delta = ((end.tv_sec - start.tv_sec) * 1000000u + end.tv_usec - start.tv_usec) / 1.e6;
                std::cout << "elapsed time is " << delta << " seconds" << std::endl;
                Rcpp::checkUserInterrupt();
            }
        }

        prob.get_param(streams, handle);

        result[lam_ind] = prob.host_B.cast<double>();
        residual_result[lam_ind] = prob.host_residual.cast<double>();
        std::cout << "Checking CUDA errors...\n";
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess)
        {
            // print the CUDA error message and exit
            std::cout << "CUDA error: " << cudaGetErrorString(error) << std::endl;
            Rcpp::stop("A CUDA error occurred during this iteration. Stop!\n");
        }
        std::cout << "None. Solution for the " << lam_ind + 1 << "th lambda pair is obtained\n";
    }
    if(nan_stop){
        terminate_nan:Rf_warning("Proximal gradient did not converge for some lambdas, return finite results only\n");
    }

    cublasDestroy(handle);
    for (uint32_t si = 0; si < K; ++si)
    {
        cudaStreamDestroy(streams[si]);
    }
    return Rcpp::List::create(Rcpp::Named("result") = result,
                              Rcpp::Named("residual") = residual_result);
}
