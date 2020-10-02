#' @export
solve_aligned_gpu <- function(X, y_list, status_list, lambda_1, lambda_2, p.fac = NULL, 
    B0 = NULL)
    {
    K <- length(y_list)
    N <- nrow(X)
    p <- ncol(X)
    status <- matrix(nrow = N, ncol = K)
    rankmin <- matrix(0L, nrow = N, ncol = K)
    rankmax <- matrix(0L, nrow = N, ncol = K)
    order_mat <- matrix(0L, nrow = N, ncol = K)
    rev_order_mat <- matrix(0L, nrow = N, ncol = K)
    for (k in 1:K)
    {
        y <- y_list[[k]]
        if (length(y) != N || length(status_list[[k]]) != N)
        {
            stop("length of y and the status must match the number of rows in X.")
        }
        o <- order(y)
        y <- y[o]
        order_mat[, k] <- o - 1L
        rev_order_mat[, k] <- order(o) - 1L
        # status[,k] = status_list[[k]][o]/N # use this vector to control the weight
        status[, k] <- status_list[[k]][o]/sum(status_list[[k]])
        rankmin[, k] <- rank(y, ties.method = "min") - 1L
        rankmax[, k] <- rank(y, ties.method = "max") - 1L
    }
    if (is.null(p.fac))
    {
        p.fac <- rep(1, p)
    }
    if (is.null(B0))
    {
        B0 <- matrix(0, p, K)
    } else
    {
        if (nrow(B0) != p || ncol(B0) < K)
        {
            stop("dimension of B is incorrect")
        }
    }
    
    solve_path(X, status, rankmin, rankmax, order_mat, rev_order_mat, B0, lambda_1, 
        lambda_2, p.fac, 1, 5000)
}

#' @export
coxph_gpu <- function(X, y, status, beta0=NULL, standardize=T)
{
    if(any(is.na(X)))
    {
        stop("We do not allow NAs in the predictor matrix")
    }

    if(any(is.na(y)))
    {
        stop("We do not allow NAs in the time vector")
    }

    if(any(is.na(status)))
    {
        stop("We do not allow NAs in the status vector")
    }

    if(standardize)
    {
        mu = apply(X, 2, mean)
        sigma  = apply(X, 2, sd)
        X = sweep(X, 2, mu, FUN='-')
        X = sweep(X, 2, sigma, FUN='/')
    }
    if(!is.null(beta0))
    {
        beta0 = as.matrix(beta0, ncol(X), 1)
    }
    result = solve_aligned_gpu(X, list(y), list(status), c(0.0), c(0.0), p.fac = NULL, 
                B0 = beta0)
    result = as.vector(result[["result"]][[1]])
    if(standardize)
    {
        result  = result / sigma
    }
    result
}