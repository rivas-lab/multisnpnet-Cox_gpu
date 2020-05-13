#ifndef PROXGPU_TYPES_H_
#define PROXGPU_TYPES_H_

#include <RcppEigen.h>

typedef Eigen::Map<Eigen::MatrixXd> MapMatd;
typedef Eigen::Map<Eigen::MatrixXf> MapMatf;
typedef Eigen::Map<Eigen::VectorXd> MapVecd;
typedef Eigen::Map<Eigen::VectorXi> MapVeci;
typedef Eigen::Map<Eigen::VectorXf> MapVecf;


typedef Eigen::MatrixXd MatrixXd;
typedef Eigen::VectorXd VectorXd;
typedef Eigen::MatrixXf MatrixXf;
typedef Eigen::VectorXf VectorXf;
typedef Eigen::VectorXi VectorXi;

typedef Eigen::PermutationMatrix<Eigen::Dynamic,Eigen::Dynamic> PermMat;


typedef float numeric;

#endif