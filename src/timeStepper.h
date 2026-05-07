#ifndef TIMESTEPPER_H
#define TIMESTEPPER_H

#include "elasticRod.h"
#include "eigenIncludes.h"
#include <vector>

#ifdef USE_MKL
#include "mkl_pardiso.h"
#include "mkl_types.h"
#include "mkl_spblas.h"
#if !defined(MKL_ILP64)
#define IFORMAT "%i"
#else
#define IFORMAT "%lli"
#endif
#else
#define IFORMAT "%i"
using MKL_INT = int;
#endif


// extern "C" void dgbsv_( int* n, int* kl, int* ku, int* nrhs, double* ab, int* ldab, int* ipiv, double* b, int* ldb, int* info );

class timeStepper
{
public:
	timeStepper(elasticRod &m_rod);
	~timeStepper();
	void setZero();
	
	void addForce(int ind, double p);
	void addJacobian(int ind1, int ind2, double p);
	void addEnergy(double p);

	void integrator();

	VectorXd getForce_py();
	MatrixXd getJacobian_py();

	void update();

	VectorXd force;
	MatrixXd jacobian;
	VectorXd dx;

	double E;

	static bool pardisoSPD(const Eigen::MatrixXd& J,
                           const Eigen::VectorXd& grad,
                           Eigen::VectorXd& dx);
		

private:
	elasticRod *rod;
	int kl, ku, freeDOF;
	
	double *totalForce;
	double *totalJacobian;


	VectorXd Force;
	MatrixXd Jacobian;


	// utility variables
	int mappedInd, mappedInd1, mappedInd2;
	int row, col, offset;
	
	int NUMROWS;
	int jacobianLen;
	int nrhs;
    int *ipiv;
    int info;
    int ldb;

	void pardisoSolver();

	struct CSR { std::vector<MKL_INT> ia, ja; std::vector<double> a; };
    static CSR buildLowerCSR(const Eigen::MatrixXd& H, double tol = 1e-14);



};


#endif
