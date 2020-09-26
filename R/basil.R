#' @importFrom mrcox basil_base
#' @export
basil <- function(genotype.pfile, phe.file, responsid, covs = NULL, nlambda = 100, 
    lambda.min.ratio = 0.01, alpha = NULL, p.factor = NULL, configs = NULL, num_lambda_per_iter = 10, 
    num_to_add = 200, max_num_to_add = 6000)
    {
    basil_base(genotype.pfile, phe.file, responsid, covs, nlambda, lambda.min.ratio, 
        alpha, p.factor, configs, num_lambda_per_iter, num_to_add, max_num_to_add, 
        solve_aligned_gpu)
}
