#include <vector>
#include <cstring>
#include <cmath>
#include <cassert>
#include <sstream>
#include "timeStepper.h"

#ifdef USE_MKL
#include <mkl.h>
#endif



timeStepper::timeStepper(elasticRod &m_rod)
{
	rod = &m_rod;
	freeDOF = rod->uncons;

	force = VectorXd::Zero(freeDOF);
	jacobian = MatrixXd::Zero(freeDOF, freeDOF);
	dx = VectorXd::Zero(freeDOF);

	Force = VectorXd::Zero(rod->ndof);
	Jacobian = MatrixXd::Zero(rod->ndof, rod->ndof);


	kl = 10; // lower diagonals
	ku = 10; // upper diagonals
	freeDOF = rod->uncons;
	ldb = freeDOF;
	NUMROWS = 2 * kl + ku + 1;
	// jacobianLen = (2 * kl + ku + 1) * freeDOF;
	// totalJacobian = new double [jacobianLen];
	nrhs = 1;
    ipiv = new int[freeDOF];
    info = 0;

}

timeStepper::~timeStepper()
{
	;
}










void timeStepper::update()
{
    freeDOF = rod->uncons;
    force = VectorXd::Zero(freeDOF);
    jacobian = MatrixXd::Zero(freeDOF, freeDOF);
    dx = VectorXd::Zero(freeDOF);
}

void timeStepper::addEnergy(double p)
{

    E += p;
}


void timeStepper::addForce(int ind, double p)
{
	if (rod->getIfConstrained(ind) == 0) // free dof
	{
		mappedInd = rod->fullToUnconsMap[ind];
		force(mappedInd) += p; // add to the force vector
	}

	Force(ind) += p;

}

void timeStepper::addJacobian(int ind1, int ind2, double p)
{
	mappedInd1 = rod->fullToUnconsMap[ind1];
	mappedInd2 = rod->fullToUnconsMap[ind2];
	if (rod->getIfConstrained(ind1) == 0 && rod->getIfConstrained(ind2) == 0) // both are free
	{
		jacobian(mappedInd2, mappedInd1) += p; // add to the jacobian

		// row = kl + ku + mappedInd2 - mappedInd1;
        // col = mappedInd1;
        // offset = row + col * NUMROWS;
        // totalJacobian[offset] += p;
	
	}
	Jacobian(ind1, ind2) += p; // add to the jacobian
}

void timeStepper::setZero()
{
	force.setZero();
	jacobian.setZero();
	Force.setZero();
	Jacobian.setZero();

    E = 0.0;

	// for (int i = 0; i < jacobianLen; i++)
	// {
	// 	totalJacobian[i] = 0;
	// }

}


void timeStepper::pardisoSolver()
{
#ifndef USE_MKL
    if (jacobian.rows() == 0)
    {
        dx.resize(0);
        return;
    }

    Eigen::FullPivLU<Eigen::MatrixXd> lu(jacobian);
    if (lu.isInvertible())
    {
        dx = lu.solve(force);
    }
    else
    {
        dx = jacobian.completeOrthogonalDecomposition().solve(force);
    }
    return;

#else
    int n = freeDOF;
    int ia[n+1];
    ia[0] = 1;

    int temp = 0;
    for (int i =0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (jacobian(i,j) != 0)
            {
                temp = temp + 1;
            }
        }
        ia[i+1] = temp+1;
    }

    int ja[ia[n]];
    double a[ia[n]];
    temp = 0;

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (jacobian(i,j) != 0)
            {
                ja[temp] = j + 1;
                a[temp] = jacobian(i,j);
                temp = temp + 1;
            }
        }
    }
    MKL_INT mtype = 11;       /* Real unsymmetric matrix */
    // Descriptor of main sparse matrix properties
    double b[n], x[n], bs[n], res, res0;
    MKL_INT nrhs = 1;     /* Number of right hand sides. */
    /* Internal solver memory pointer pt, */
    /* 32-bit: int pt[64]; 64-bit: long int pt[64] */
    /* or void *pt[64] should be OK on both architectures */
    void *pt[64];
    /* Pardiso control parameters. */
    MKL_INT iparm[64];
    MKL_INT maxfct, mnum, phase, error, msglvl;
    /* Auxiliary variables. */
    MKL_INT i, j;
    double ddum;          /* Double dummy */
    MKL_INT idum;         /* Integer dummy. */
/* -------------------------------------------------------------------- */
/* .. Setup Pardiso control parameters. */
/* -------------------------------------------------------------------- */
    for ( i = 0; i < 64; i++ )
    {
        iparm[i] = 0;
    }
    iparm[0] = 0;         /* No solver default */
    iparm[1] = 2;         /* Fill-in reordering from METIS */
    iparm[3] = 0;         /* No iterative-direct algorithm */
    iparm[4] = 0;         /* No user fill-in reducing permutation */
    iparm[5] = 0;         /* Write solution into x */
    iparm[6] = 0;         /* Not in use */
    iparm[7] = 2;         /* Max numbers of iterative refinement steps */
    iparm[8] = 0;         /* Not in use */
    iparm[9] = 13;        /* Perturb the pivot elements with 1E-13 */
    iparm[10] = 1;        /* Use nonsymmetric permutation and scaling MPS */
    iparm[11] = 0;        /* Conjugate transposed/transpose solve */
    iparm[12] = 1;        /* Maximum weighted matching algorithm is switched-on (default for non-symmetric) */
    iparm[13] = 0;        /* Output: Number of perturbed pivots */
    iparm[14] = 0;        /* Not in use */
    iparm[15] = 0;        /* Not in use */
    iparm[16] = 0;        /* Not in use */
    iparm[17] = -1;       /* Output: Number of nonzeros in the factor LU */
    iparm[18] = -1;       /* Output: Mflops for LU factorization */
    iparm[19] = 0;        /* Output: Numbers of CG Iterations */


    maxfct = 1;           /* Maximum number of numerical factorizations. */
    mnum = 1;         /* Which factorization to use. */
    msglvl = 0;           /* Print statistical information  */
    error = 0;            /* Initialize error flag */
/* -------------------------------------------------------------------- */
/* .. Initialize the internal solver memory pointer. This is only */
/* necessary for the FIRST call of the PARDISO solver. */
/* -------------------------------------------------------------------- */
    for ( i = 0; i < 64; i++ )
    {
        pt[i] = 0;
    }
/* -------------------------------------------------------------------- */
/* .. Reordering and Symbolic Factorization. This step also allocates */
/* all memory that is necessary for the factorization. */
/* -------------------------------------------------------------------- */
    phase = 11;
    PARDISO (pt, &maxfct, &mnum, &mtype, &phase,
             &n, a, ia, ja, &idum, &nrhs, iparm, &msglvl, &ddum, &ddum, &error);
    if ( error != 0 )
    {
        printf ("\nERROR during symbolic factorization: " IFORMAT, error);
        exit (1);
    }
    // printf ("\nReordering completed ... ");
    // printf ("\nNumber of nonzeros in factors = " IFORMAT, iparm[17]);
    // printf ("\nNumber of factorization MFLOPS = " IFORMAT, iparm[18]);
/* -------------------------------------------------------------------- */
/* .. Numerical factorization. */
/* -------------------------------------------------------------------- */
    phase = 22;
    PARDISO (pt, &maxfct, &mnum, &mtype, &phase,
             &n, a, ia, ja, &idum, &nrhs, iparm, &msglvl, &ddum, &ddum, &error);
    if ( error != 0 )
    {
        printf ("\nERROR during numerical factorization: " IFORMAT, error);
        exit (2);
    }
    // printf ("\nFactorization completed ... ");
/* -------------------------------------------------------------------- */
/* .. Back substitution and iterative refinement. */
/* -------------------------------------------------------------------- */
    phase = 33;

// descrA.type = SPARSE_MATRIX_TYPE_GENERAL;
// descrA.mode = SPARSE_FILL_MODE_UPPER;
// descrA.diag = SPARSE_DIAG_NON_UNIT;
// mkl_sparse_d_create_csr ( &csrA, SPARSE_INDEX_BASE_ONE, n, n, ia, ia+1, ja, a );

    /* Set right hand side to one. */
    for ( i = 0; i < n; i++ )
    {
        b[i] = force[i];
    }
//  Loop over 3 solving steps: Ax=b, AHx=b and ATx=b
    PARDISO (pt, &maxfct, &mnum, &mtype, &phase,
             &n, a, ia, ja, &idum, &nrhs, iparm, &msglvl, b, x, &error);
    if ( error != 0 )
    {
        printf ("\nERROR during solution: " IFORMAT, error);
        exit (3);
    }

    // printf ("\nThe solution of the system is: ");
    // for ( j = 0; j < n; j++ )
    // {
    //     printf ("\n x [" IFORMAT "] = % f", j, x[j]);
    // }
    // printf ("\n");

/* -------------------------------------------------------------------- */
/* .. Termination and release of memory. */
/* -------------------------------------------------------------------- */
    phase = -1;           /* Release internal memory. */
    PARDISO (pt, &maxfct, &mnum, &mtype, &phase,
             &n, &ddum, ia, ja, &idum, &nrhs,
             iparm, &msglvl, &ddum, &ddum, &error);

    for (int i = 0; i < n; i++)
    {
        dx[i] = x[i];
    }



    // auto stop = high_resolution_clock::now();
    // auto duration = duration_cast<microseconds>(stop - start);
    //
    // cout << "Time taken by function: "
    // 		 << duration.count() << " microseconds" << endl;
#endif
}



void timeStepper::integrator()
{
	pardisoSolver();
	// dgbsv_(&freeDOF, &kl, &ku, &nrhs, totalJacobian, &NUMROWS, ipiv, totalForce, &ldb, &info);

	// for (int i = 0; i < freeDOF; i++)
	// {
	// 	cout << dx[i] << " " <<totalForce[i] << endl;
	// }
	// exit(0);
}

VectorXd timeStepper::getForce_py()
{
	VectorXd result(2 * rod->nv);

	for (int i = 0; i < rod->nv; i++)
	{
		result(2 * i) = Force(4 * i);
		result(2 * i + 1) = Force(4 * i + 2);
	}


	return result;
}

MatrixXd timeStepper::getJacobian_py()
{
	MatrixXd result(2 * rod->nv, 2 * rod->nv);

	for (int i = 0; i < rod->nv; i++)
	{
		for (int j = 0; j < rod->nv; j++)
		{
			result(2 * i, 2 * j) = Jacobian(4 * i, 4 * j);
			result(2 * i + 1, 2 * j + 1) = Jacobian(4 * i + 2, 4 * j + 2);
			result(2 * i, 2 * j + 1) = Jacobian(4 * i, 4 * j + 2);
			result(2 * i + 1, 2 * j) = Jacobian(4 * i + 2, 4 * j);
		}
	}


	return result;
}


// bool timeStepper::pardisoSPD(const Eigen::MatrixXd& J,
//                             const Eigen::VectorXd& grad,
//                             Eigen::VectorXd& dxx)
// {
  
//    int n = J.rows();


//    int ia[n+1];
//    ia[0] = 1;


//    int temp = 0;
//    for (int i =0; i < n; i++)
//    {
//        for (int j = 0; j < n; j++)
//        {
//            if (J(i,j) != 0)
//            {
//                temp = temp + 1;
//            }
//        }
//        ia[i+1] = temp+1;
//    }


//    int ja[ia[n]];
//    double a[ia[n]];
//    temp = 0;


//    for (int i = 0; i < n; i++)
//    {
//        for (int j = 0; j < n; j++)
//        {
//            if (J(i,j) != 0)
//            {
//                ja[temp] = j + 1;
//                a[temp] = J(i,j);
//                temp = temp + 1;
//            }
//        }
//    }
//    MKL_INT mtype = 11;       /* Real Symmetric matrix */
//    // Descriptor of main sparse matrix properties
//    double b[n], x[n], bs[n], res, res0;
//    MKL_INT nrhs = 1;     /* Number of right hand sides. */
//    /* Internal solver memory pointer pt, */
//    /* 32-bit: int pt[64]; 64-bit: long int pt[64] */
//    /* or void *pt[64] should be OK on both architectures */
//    void *pt[64];
//    /* Pardiso control parameters. */
//    MKL_INT iparm[64];
//    MKL_INT maxfct, mnum, phase, error, msglvl;
//    /* Auxiliary variables. */
//    MKL_INT i, j;
//    double ddum;          /* Double dummy */
//    MKL_INT idum;         /* Integer dummy. */
// /* -------------------------------------------------------------------- */
// /* .. Setup Pardiso control parameters. */
// /* -------------------------------------------------------------------- */
//    for ( i = 0; i < 64; i++ )
//    {
//        iparm[i] = 0;
//    }
//    iparm[0] = 0;         /* No solver default */
//    iparm[1] = 2;         /* Fill-in reordering from METIS */
//    iparm[3] = 0;         /* No iterative-direct algorithm */
//    iparm[4] = 0;         /* No user fill-in reducing permutation */
//    iparm[5] = 0;         /* Write solution into x */
//    iparm[6] = 0;         /* Not in use */
//    iparm[7] = 2;         /* Max numbers of iterative refinement steps */
//    iparm[8] = 0;         /* Not in use */
//    iparm[9] = 13;        /* Perturb the pivot elements with 1E-13 */
//    iparm[10] = 1;        /* Use nonsymmetric permutation and scaling MPS */
//    iparm[11] = 0;        /* Conjugate transposed/transpose solve */
//    iparm[12] = 1;        /* Maximum weighted matching algorithm is switched-on (default for non-symmetric) */
//    iparm[13] = 0;        /* Output: Number of perturbed pivots */
//    iparm[14] = 0;        /* Not in use */
//    iparm[15] = 0;        /* Not in use */
//    iparm[16] = 0;        /* Not in use */
//    iparm[17] = -1;       /* Output: Number of nonzeros in the factor LU */
//    iparm[18] = -1;       /* Output: Mflops for LU factorization */
//    iparm[19] = 0;        /* Output: Numbers of CG Iterations */




//    maxfct = 1;           /* Maximum number of numerical factorizations. */
//    mnum = 1;         /* Which factorization to use. */
//    msglvl = 0;           /* Print statistical information  */
//    error = 0;            /* Initialize error flag */
// /* -------------------------------------------------------------------- */
// /* .. Initialize the internal solver memory pointer. This is only */
// /* necessary for the FIRST call of the PARDISO solver. */
// /* -------------------------------------------------------------------- */
//    for ( i = 0; i < 64; i++ )
//    {
//        pt[i] = 0;
//    }
// /* -------------------------------------------------------------------- */
// /* .. Reordering and Symbolic Factorization. This step also allocates */
// /* all memory that is necessary for the factorization. */
// /* -------------------------------------------------------------------- */
//    phase = 11;
//    PARDISO (pt, &maxfct, &mnum, &mtype, &phase,
//             &n, a, ia, ja, &idum, &nrhs, iparm, &msglvl, &ddum, &ddum, &error);
//    if ( error != 0 )
//    {
//        printf ("\nERROR during symbolic factorization: " IFORMAT, error);
//        exit (1);
//    }
//    // printf ("\nReordering completed ... ");
//    // printf ("\nNumber of nonzeros in factors = " IFORMAT, iparm[17]);
//    // printf ("\nNumber of factorization MFLOPS = " IFORMAT, iparm[18]);
// /* -------------------------------------------------------------------- */
// /* .. Numerical factorization. */
// /* -------------------------------------------------------------------- */
//    phase = 22;
//    PARDISO (pt, &maxfct, &mnum, &mtype, &phase,
//             &n, a, ia, ja, &idum, &nrhs, iparm, &msglvl, &ddum, &ddum, &error);
//    if ( error != 0 )
//    {
//        printf ("\nERROR during numerical factorization: " IFORMAT, error);
//        exit (2);
//    }
//    // printf ("\nFactorization completed ... ");
// /* -------------------------------------------------------------------- */
// /* .. Back substitution and iterative refinement. */
// /* -------------------------------------------------------------------- */
//    phase = 33;


// // descrA.type = SPARSE_MATRIX_TYPE_GENERAL;
// // descrA.mode = SPARSE_FILL_MODE_UPPER;
// // descrA.diag = SPARSE_DIAG_NON_UNIT;
// // mkl_sparse_d_create_csr ( &csrA, SPARSE_INDEX_BASE_ONE, n, n, ia, ia+1, ja, a );


//    /* Set right hand side to one. */
//    for ( i = 0; i < n; i++ )
//    {
//        b[i] = grad[i];
//    }
// //  Loop over 3 solving steps: Ax=b, AHx=b and ATx=b
//    PARDISO (pt, &maxfct, &mnum, &mtype, &phase,
//             &n, a, ia, ja, &idum, &nrhs, iparm, &msglvl, b, x, &error);
//    if ( error != 0 )
//    {
//        printf ("\nERROR during solution: " IFORMAT, error);
//        exit (3);
//    }


// /* -------------------------------------------------------------------- */
// /* .. Termination and release of memory. */
// /* -------------------------------------------------------------------- */
//    phase = -1;           /* Release internal memory. */
//    PARDISO (pt, &maxfct, &mnum, &mtype, &phase,
//             &n, &ddum, ia, ja, &idum, &nrhs,
//             iparm, &msglvl, &ddum, &ddum, &error);

//    dxx.resize(n);
//    for (int i = 0; i < n; i++)
//    {
//        dxx[i] = x[i];
//    }


//    return true;
// }

// Build LOWER-triangular CSR (1-based) from a dense symmetric J
static void build_lower_csr(const Eigen::MatrixXd& J,
                            std::vector<MKL_INT>& ia,
                            std::vector<MKL_INT>& ja,
                            std::vector<double>&  a,
                            double tol = 1e-14)
{
    const MKL_INT n = static_cast<MKL_INT>(J.rows());
    ia.assign(n+1, 0);
    ja.clear(); a.clear();
    ia[0] = 1;                       // 1-based CSR
    MKL_INT nnz = 0;
    for (MKL_INT i=0;i<n;++i) {
        for (MKL_INT j=0;j<=i;++j) { // LOWER ONLY
            double v = 0.5*(J(i,j)+J(j,i));  // numeric symmetrize
            if (std::abs(v) > tol) {
                ja.push_back(j+1);           // 1-based col
                a .push_back(v);
                ++nnz;
            }
        }
        ia[i+1] = nnz + 1;                   // next row start (1-based)
    }
    // quick checks
    assert(ia.back() == static_cast<MKL_INT>(a.size()) + 1);
}


static void build_upper_csr(const Eigen::MatrixXd& J,
                            std::vector<MKL_INT>& ia,
                            std::vector<MKL_INT>& ja,
                            std::vector<double>&  a,
                            double tol = 1e-14)
{
    const MKL_INT n = static_cast<MKL_INT>(J.rows());
    ia.assign(n+1, 0);
    ja.clear(); a.clear();
    ia[0] = 1;                    // 1-based CSR
    MKL_INT nnz = 0;

    for (MKL_INT i = 0; i < n; ++i) {
        for (MKL_INT j = i; j < n; ++j) {         // UPPER ONLY (j >= i)
            const double v = 0.5*(J(i,j) + J(j,i));   // numeric symmetrize
            if (std::abs(v) > tol) {
                ja.push_back(j + 1);              // 1-based col index
                a .push_back(v);
                ++nnz;
            }
        }
        ia[i+1] = nnz + 1;                        // next row start (1-based)
    }

    // quick invariants
    if (ia.back() != static_cast<MKL_INT>(a.size()) + 1) {
        throw std::runtime_error("CSR size mismatch");
    }

    // debug: verify each row has only col >= row (upper)
    for (MKL_INT i = 0; i < n; ++i) {
        for (MKL_INT k = ia[i]-1; k < ia[i+1]-1; ++k) {
            if (ja[k] < i+1) {
                std::ostringstream oss;
                oss << "Upper-CSR violation at row " << (i+1)
                    << " col " << ja[k];
                throw std::runtime_error(oss.str());
            }
        }
    }
}


bool timeStepper::pardisoSPD(const Eigen::MatrixXd& J,
                             const Eigen::VectorXd& grad,
                             Eigen::VectorXd& dxx)
{
#ifndef USE_MKL
    if (J.rows() == 0 || J.cols() != J.rows() || grad.size() != J.rows())
    {
        return false;
    }

    Eigen::LDLT<Eigen::MatrixXd> ldlt(0.5 * (J + J.transpose()));
    if (ldlt.info() == Eigen::Success && ldlt.isPositive())
    {
        dxx = ldlt.solve(-grad);
        return dxx.allFinite();
    }

    Eigen::FullPivLU<Eigen::MatrixXd> lu(J);
    if (lu.isInvertible())
    {
        dxx = lu.solve(-grad);
        return dxx.allFinite();
    }
    return false;

#else
    const MKL_INT n = static_cast<MKL_INT>(J.rows());
    if (J.cols()!=n || grad.size()!=n) return false;

    std::vector<MKL_INT> ia, ja;
    std::vector<double>  a;
    build_upper_csr(J, ia, ja, a, 1e-14);   // <-- UPPER triangle only

    void*   pt[64] = {nullptr};
    MKL_INT iparm[64] = {0};
    iparm[0]  = 1;   // honor iparm[]
    iparm[1]  = 2;   // METIS
    iparm[2]  = 1;   // 1 thread while debugging
    iparm[7]  = 2;   // iterative refinement
    iparm[9]  = 13;  // pivot perturbation
    iparm[10] = 1;   // scaling/permutation
    // iparm[26] = 1;   // matrix checker
    int msglvl = 0;

    MKL_INT maxfct=1, mnum=1, nrhs=1, phase, error=0;
    MKL_INT* perm = nullptr;        // IMPORTANT: NULL perm
    double ddum = 0.0;
    MKL_INT mtype = 2;              // SPD

    // 11) analyze
    phase = 11;
    PARDISO(pt,&maxfct,&mnum,&mtype,&phase,&n,
            a.data(), ia.data(), ja.data(), perm,&nrhs,
            iparm,&msglvl, &ddum, &ddum, &error);
    if (error) { phase=-1; PARDISO(pt,&maxfct,&mnum,&mtype,&phase,&n,
                    &ddum, ia.data(), ja.data(), perm,&nrhs,
                    iparm,&msglvl, &ddum, &ddum, &error); return false; }

    // 22) factor
    phase = 22;
    PARDISO(pt,&maxfct,&mnum,&mtype,&phase,&n,
            a.data(), ia.data(), ja.data(), perm,&nrhs,
            iparm,&msglvl, &ddum, &ddum, &error);
    if (error) { phase=-1; PARDISO(pt,&maxfct,&mnum,&mtype,&phase,&n,
                    &ddum, ia.data(), ja.data(), perm,&nrhs,
                    iparm,&msglvl, &ddum, &ddum, &error); return false; }

    // 33) solve: J d = -grad (Newton)
    std::vector<double> b(n), x(n);
    for (MKL_INT i=0;i<n;++i) b[i] = -grad[i];

    phase = 33;
    PARDISO(pt,&maxfct,&mnum,&mtype,&phase,&n,
            a.data(), ia.data(), ja.data(), perm,&nrhs,
            iparm,&msglvl, b.data(), x.data(), &error);

    phase = -1; PARDISO(pt,&maxfct,&mnum,&mtype,&phase,&n,
                        &ddum, ia.data(), ja.data(), perm,&nrhs,
                        iparm,&msglvl, &ddum, &ddum, &error);

    if (error) return false;

    dxx.resize(n);
    for (MKL_INT i=0;i<n;++i) dxx[i] = x[i];
    return true;
#endif
}


// bool timeStepper::pardisoSPD(const Eigen::MatrixXd& J,
//                              const Eigen::VectorXd& grad,
//                              Eigen::VectorXd& dxx)
// {
//     const MKL_INT n = static_cast<MKL_INT>(J.rows());
//     if (J.cols()!=n || grad.size()!=n) return false;

//     std::vector<MKL_INT> ia, ja;
//     std::vector<double>  a;
//     build_lower_csr(J, ia, ja, a, 1e-14);     // LOWER only, 1-based

//     void*   pt[64] = {nullptr};
//     MKL_INT iparm[64] = {0};
//     iparm[0]=1; iparm[1]=2; iparm[2]=1;       // 1 thread while debugging
//     iparm[7]=2; iparm[9]=13; iparm[10]=1;
//     iparm[26]=1;                               // matrix checker on
//     int msglvl = 1;

//     MKL_INT maxfct=1, mnum=1, nrhs=1, phase, error=0;
//     MKL_INT* perm = nullptr; double ddum=0.0;
//     MKL_INT mtype = 2;                         // SPD

//     std::cout << "n=" << n
//           << " nnz=" << (ia.back()-1)
//           << " ia[0]=" << ia[0]
//           << " ia[n]=" << ia.back()
//           << std::endl;

// // Check row 194 specifically (1-based -> 0-based = 193):
// int r = 193;
// for (MKL_INT k = ia[r]-1; k < ia[r+1]-1; ++k) {
//     std::cout << "row " << (r+1) << " col=" << ja[k] << "\n"; // must be >= r+1
// }

//     // 11) analyze
//     phase = 11;
//     PARDISO(pt,&maxfct,&mnum,&mtype,&phase,&n,
//             a.data(), ia.data(), ja.data(), perm,&nrhs,
//             iparm,&msglvl, &ddum, &ddum, &error);
//     if (error) { phase=-1; PARDISO(pt,&maxfct,&mnum,&mtype,&phase,&n,
//                     &ddum, ia.data(), ja.data(), perm,&nrhs,
//                     iparm,&msglvl, &ddum, &ddum, &error); return false; }

//     // 22) factor
//     phase = 22;
//     PARDISO(pt,&maxfct,&mnum,&mtype,&phase,&n,
//             a.data(), ia.data(), ja.data(), perm,&nrhs,
//             iparm,&msglvl, &ddum, &ddum, &error);
//     if (error) { phase=-1; PARDISO(pt,&maxfct,&mnum,&mtype,&phase,&n,
//                     &ddum, ia.data(), ja.data(), perm,&nrhs,
//                     iparm,&msglvl, &ddum, &ddum, &error); return false; }

//     // 33) solve: J d = -grad
//     std::vector<double> b(n), x(n);
//     for (MKL_INT i=0;i<n;++i) b[i] = -grad[i];
//     phase = 33;
//     PARDISO(pt,&maxfct,&mnum,&mtype,&phase,&n,
//             a.data(), ia.data(), ja.data(), perm,&nrhs,
//             iparm,&msglvl, b.data(), x.data(), &error);

//     phase = -1; // release
//     PARDISO(pt,&maxfct,&mnum,&mtype,&phase,&n,
//             &ddum, ia.data(), ja.data(), perm,&nrhs,
//             iparm,&msglvl, &ddum, &ddum, &error);

//     if (error) return false;

//     dxx = Eigen::Map<Eigen::VectorXd>(x.data(), n);
//     return true;
// }
