#' @export
solve_aligned_gpu = function(X, y_list, status_list, lambda_1, lambda_2, p.fac=NULL, B0=NULL){
  K = length(y_list)
  N = nrow(X)
  p = ncol(X)
  status = matrix(nrow=N, ncol=K)
  rankmin = matrix(0L, nrow=N, ncol=K)
  rankmax = matrix(0L, nrow=N, ncol=K)
  order_mat = matrix(0L, nrow=N, ncol=K)
  rev_order_mat = matrix(0L, nrow=N, ncol=K)
  for(k in 1:K){
    y = y_list[[k]]
    if(length(y) != N || length(status_list[[k]]) != N){
      stop("length of y and the status must match the number of rows in X.")
    }
    o = order(y)
    y = y[o]
    order_mat[,k] = o - 1L
    rev_order_mat[,k] = order(o) - 1L
    #status[,k] = status_list[[k]][o]/N # use this vector to control the weight
    status[,k] = status_list[[k]][o]/sum(status_list[[k]])
    rankmin[,k] = rank(y, ties.method="min") - 1L
    rankmax[,k] = rank(y, ties.method="max") - 1L    
  }
  if(is.null(p.fac)){
    p.fac = rep(1.0, p)
  }
  if(is.null(B0)){
    B0 = matrix(0.0, p, K)
  } else {
    if(nrow(B0)!= p || ncol(B0) < K){
      stop("dimension of B is incorrect")
    }
  }
  
   solve_path(X,
               status,
               rankmin,
               rankmax,
               order_mat,
               rev_order_mat,
               B0,
               lambda_1,
               lambda_2,
               p.fac,
              1.0,
              5000)
}