#include "solver.h"
#include <algorithm>
#include <cmath>
#include <chrono>

#include "elasticRod.h"   // include your rod interface
#include "timeStepper.h" // include your time stepper interface
#include "elasticStretchingForce.h" // include your stretching force interface
#include "elasticBendingForce.h"   // include your bending force interface
#include "externalGravityForce.h"  // include your gravity force interface

using Vec = Eigen::VectorXd;
using Mat = Eigen::MatrixXd;
using SpMat = Eigen::SparseMatrix<double, Eigen::ColMajor>;

bool isSPD(const Eigen::MatrixXd& H,
           double sym_tol = 1e-12,
           double pos_tol = 1e-14)
{
    const int n = H.rows();
    if (n == 0 || H.cols() != n) return false;
    if (!H.allFinite())          return false;

    // 1) symmetry (relative) check
    const double nrm = H.norm();
    const double rel = sym_tol * std::max(1.0, nrm);
    if (!H.isApprox(H.transpose(), rel)) return false;

    // 2) work with a numerically symmetrized copy
    Eigen::MatrixXd S = 0.5 * (H + H.transpose());

    // 3) try Cholesky (fast path)
    Eigen::LLT<Eigen::MatrixXd> llt(S);
    if (llt.info() == Eigen::Success) return true;

    // 4) fall back to LDLT (with pivoting) and check signs in D
    Eigen::LDLT<Eigen::MatrixXd> ldlt(S);
    if (ldlt.info() != Eigen::Success) return false;

    // Some Eigen versions have ldlt.isPositive(); if so, you can just return that.
    // Here we check D's diagonal against a scale-aware tolerance.
    const double scale = S.diagonal().cwiseAbs().maxCoeff();
    const double tol   = pos_tol * std::max(1.0, scale);
    const auto   D     = ldlt.vectorD();
    return (D.array() > tol).all();
}


// --------------------------------
solver::solver(elasticRod &m_rod, timeStepper &m_stepper, 
               elasticStretchingForce &m_stretchForce, 
               elasticBendingForce &m_bendingForce, 
               externalGravityForce &m_gravityForce) : 
               rod(&m_rod), stepper(&m_stepper), 
               m_stretchForce(&m_stretchForce), 
               m_bendingForce(&m_bendingForce), 
               m_gravityForce(&m_gravityForce) {

    freeDOF = rod->uncons;
    
    // define the forces and jacobians
    force = Vec::Zero(freeDOF);
    jacobian = Mat::Zero(freeDOF, freeDOF);
    dx = Vec::Zero(freeDOF);
}

solver::~solver(){
    ;
}


Eigen::VectorXd solver::stepToBoundary_(const Vec &p, const Vec &d, double Delta){
    const double a = d.squaredNorm();
    const double b = 2.0 * p.dot(d);
    const double c = p.squaredNorm() - Delta*Delta;
    const double disc = std::max(0.0, b*b - 4.0*a*c);
    const double tau  = (-b + std::sqrt(disc)) / (2.0*a); // outer root
    return p + tau * d;
} 

Eigen::VectorXd solver::jacobiSolve_(const Mat& H, const Vec& r, double shift) {
    Vec invd = H.diagonal().cwiseAbs().array() + shift;
    invd = invd.cwiseInverse();
    return invd.asDiagonal() * r;
}

// -------------------CG-----------------------
Eigen::VectorXd solver::steihaugPCG(const Vec &g,
                                    const std::function<Vec(const Vec &)> &Hv,  
                                    const std::function<Vec(const Vec &)> &Msolve,
                                    double Delta, double tol) const{
    Vec p = Vec::Zero(g.size()); // gradient size
    if (g.norm() == 0.0 || Delta <= 0.0) return p;

    Vec r = g;        // initial residual 
    Vec z = Msolve(r); // preconditioned residual  
    Vec d = -z;
    double rz = r.dot(z);
    int maxit = 200;

    if (r.norm() < tol) return p;

    for (int k = 0; k < maxit; k++)
    {

        Vec Hd = Hv(d);
        const double dHd = d.dot(Hd);

        if (dHd <= 0) { // negative curvature
            return stepToBoundary_(p, d, Delta);
        }

        const double alpha = rz / dHd;
        Vec p_trial = p + alpha * d;

        if (p_trial.norm() >= Delta)
            return stepToBoundary_(p, d, Delta);   // hit TR boundary

        p = p_trial;
        r = r + alpha * Hd;
        const double rnorm = r.norm();
        if (rnorm <= tol) 
            return p;   // converged

        Vec z_new = Msolve(r);
        const double rz_new = r.dot(z_new);


       const double beta = rz_new / rz;
       d = -z_new + beta * d;
       z = z_new;
       rz = rz_new;
    }

    return p;
}

// ---------- assembleAt(x_free) ----------
// Replace the placeholder with your real rod calls to fill:
//   • force (reduced grad), jacobian (reduced Hessian), and return E(x_free)
double solver::assembleAt(const Vec& x_free)
{
    rod->setFreeDOF(x_free);

    rod->prepareForIteration();
    stepper->setZero();

    m_stretchForce->computeFs();
    m_stretchForce->computeJs();

    m_bendingForce->computeFb();
    m_bendingForce->computeJb();

    m_gravityForce->computeFg();
    m_gravityForce->computeJg();

    force = stepper->force; // this is just \nabla E
    jacobian = stepper->jacobian; // this is just \nabla^2



    return stepper->E;
}


// ----------- TR outer loop -----------
Eigen::VectorXd solver::minimizeEnergy_TR(const Vec &x_free0, const TROpts &opt)
{
    Vec x = x_free0;
    double Delta = opt.Delta0;

    using clock = std::chrono::high_resolution_clock;

    int solver_iter = 0;
    for (int k = 0; k < opt.max_outer; k++)
    {
        auto t1 = clock::now();
        // 1) Assemble energy, grad, Hessian at current x
        double E_old = assembleAt(x);        
        Vec g = force; // gradient
        const double gnorm = g.norm();

        // cout << "E_old: " << E_old << endl; 
        // cout << "gnorm: " << gnorm << endl;
        // cout << "Delta: " << Delta << endl;

        // cout << "iter: " << k << " gradient: " << gnorm << endl;
        if (gnorm < opt.tol_g) 
        {
            break;
        }

        Mat H = 0.5 * (jacobian + jacobian.transpose()); // numeric symmetrize

        // check the direct solve
        Vec pN, p;
        const bool spd_ok = stepper->pardisoSPD(H, g, pN);
        
        const double tol_cg = std::min(0.5, std::sqrt(gnorm)) * gnorm;

        // 2) freeze H for this subproblem and define Hv + preconditioner
        std::function<Vec(const Vec&)> Hv;
        std::function<Vec(const Vec&)> Msolve;

        Hv = [H](const Vec &v) -> Vec { return H * v; }; // dense
        Msolve = [H](const Vec &r) -> Vec { return jacobiSolve_(H, r); };

        if (spd_ok) {
            if (pN.norm() <= Delta) { // full Newton step is ok
                p = pN; 
            } else {
                Vec Hg = Hv(g);
                double gHg = g.dot(Hg);

                Vec pC;
                const double eps = 1e-12;

                if (gHg <= eps * gnorm * gnorm) {
                    pC = - (Delta / std::max(gnorm, 1e-16)) * g; // scaled gradient
                } else
                {
                    double alphaC = (gnorm*gnorm) / gHg;
                    pC = - alphaC * g; // scale gradient to make 
                }


                if (pC.norm() >= Delta) {
                    p = - (Delta / std::max(gnorm, 1e-16)) * g; // Cauchy point
                } else {                
                    // intersect the segment pC -> pN with the boundary
                    Vec dseg = pN - pC;
                    double a = dseg.squaredNorm();
                    double b = 2.0 * pC.dot(dseg);
                    double c = pC.squaredNorm() - Delta*Delta;

                    if (a <= 1e-30) {
                        return pC;
                    }
                    double disc = b * b - 4 * a * c;
                    if (disc < 0) disc = 0.0;
                    double tau = (-b + std::sqrt(disc)) / (2*a);  // in (0,1]
                    tau = std::min(std::max(tau, 0.0), 1.0);
                    p = pC + tau * dseg; 
                }
            }
        }
        else{
            // cout <<"using cg" << endl;
            p = steihaugPCG(g, Hv, Msolve, Delta, tol_cg); // step
        }

        // 4) predicted reduction
        Vec Hp = Hv(p);
        double pred = -(g.dot(p)) - 0.5 * p.dot(Hp);

        if (pred <= 0.0) {
            Vec p_sd = - std::min(Delta / std::max(gnorm, 1e-16), 1.0) * g;
            Vec Hpsd = Hv(p_sd);
            double pred_sd = -(g.dot(p_sd)) - 0.5 * p_sd.dot(Hpsd);
            
            if (pred_sd > pred) { p = p_sd; pred = pred_sd; }
            // p    = p_sd;
            // pred = std::max(pred, pred_sd);   
        }

        // 5) actual reduction
        double E_new = assembleAt(x + p);
        double rho   = (E_old - E_new) / std::max(pred, 1e-16);

        // 6) update radius
        if      (rho < 0.25) Delta *= opt.shrink;
        else if (rho > 0.75 && std::abs(p.norm() - Delta) / std::max(Delta,1e-16) < 1e-6)
                             Delta *= opt.expand;

        solver_iter++;
        // if (Delta < 1e-7) Delta = 1e-7;
        if (Delta <= 1e-8)
        {
            break;
        }
        // cout << "iter: " << k  << "Delta : " << Delta << " rho: " << rho << " |g|: " << gnorm << " |p|: " << p.norm() << " E: " << E_new << endl;


        // 7) accept?
        if (rho >= opt.eta) x += p;

        // if (solver_iter > 100)
        // {
        //     cout << "Too many iterations" << endl;
        //     exit(0);
        //     break;
        // }

        // cout << "Elapsed time: " 
        //      << std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - t1).count() 
        //      << " ms" << endl;
    }

    // exit(0);
    // cout <<"solver iter: " << solver_iter << endl;
    // if (solver_iter > 200)
    // {
    //     exit(0);
    // }

    return x;
}
