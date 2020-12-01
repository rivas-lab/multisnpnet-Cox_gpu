// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include "proxgpu_types.h"
#include <RcppEigen.h>
#include <Rcpp.h>

using namespace Rcpp;

// solve_path
Rcpp::List solve_path(Rcpp::NumericMatrix X, Rcpp::NumericMatrix status, Rcpp::IntegerMatrix rankmin, Rcpp::IntegerMatrix rankmax, Rcpp::IntegerMatrix order, Rcpp::IntegerMatrix rev_order, Rcpp::NumericMatrix B0, Rcpp::NumericVector lambda_1_all, Rcpp::NumericVector lambda_2_all, Rcpp::NumericVector pfac, double step_sized, int niter, double linesearch_betad);
RcppExport SEXP _proxgpu_solve_path(SEXP XSEXP, SEXP statusSEXP, SEXP rankminSEXP, SEXP rankmaxSEXP, SEXP orderSEXP, SEXP rev_orderSEXP, SEXP B0SEXP, SEXP lambda_1_allSEXP, SEXP lambda_2_allSEXP, SEXP pfacSEXP, SEXP step_sizedSEXP, SEXP niterSEXP, SEXP linesearch_betadSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type X(XSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type status(statusSEXP);
    Rcpp::traits::input_parameter< Rcpp::IntegerMatrix >::type rankmin(rankminSEXP);
    Rcpp::traits::input_parameter< Rcpp::IntegerMatrix >::type rankmax(rankmaxSEXP);
    Rcpp::traits::input_parameter< Rcpp::IntegerMatrix >::type order(orderSEXP);
    Rcpp::traits::input_parameter< Rcpp::IntegerMatrix >::type rev_order(rev_orderSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericMatrix >::type B0(B0SEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type lambda_1_all(lambda_1_allSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type lambda_2_all(lambda_2_allSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type pfac(pfacSEXP);
    Rcpp::traits::input_parameter< double >::type step_sized(step_sizedSEXP);
    Rcpp::traits::input_parameter< int >::type niter(niterSEXP);
    Rcpp::traits::input_parameter< double >::type linesearch_betad(linesearch_betadSEXP);
    rcpp_result_gen = Rcpp::wrap(solve_path(X, status, rankmin, rankmax, order, rev_order, B0, lambda_1_all, lambda_2_all, pfac, step_sized, niter, linesearch_betad));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_proxgpu_solve_path", (DL_FUNC) &_proxgpu_solve_path, 13},
    {NULL, NULL, 0}
};

RcppExport void R_init_proxgpu(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
