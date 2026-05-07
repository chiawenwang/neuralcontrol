#pragma once
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <functional>

class elasticRod;
class timeStepper;
class elasticStretchingForce;
class elasticBendingForce;
class externalGravityForce;

class solver{
public:
    explicit solver(elasticRod &m_rod, timeStepper &m_stepper, 
                    elasticStretchingForce &m_stretchForce, 
                    elasticBendingForce &m_bendingForce, 
                    externalGravityForce &m_gravityForce);
    ~solver();

    struct TROpts{
        double Delta0 = 1e-3;
        double eta = 0.1;
        double shrink = 0.5;
        double expand = 2.0;
        double tol_g = 1e-9;
        int max_outer = 200;
    };

    Eigen::VectorXd minimizeEnergy_TR(const Eigen::VectorXd &x_free0, 
                                      const TROpts& opt);

    double assembleAt(const Eigen::VectorXd &x_free);

    // steihaug PCG (inner TR solver)
    Eigen::VectorXd steihaugPCG(const Eigen::VectorXd &g,
                                const std::function<Eigen::VectorXd(const Eigen::VectorXd &)> &Hv,
                                const std::function<Eigen::VectorXd(const Eigen::VectorXd &)> &Msolve,
                                double Delta, double tol) const;
    
    static Eigen::VectorXd stepToBoundary_(const Eigen::VectorXd &p, const Eigen::VectorXd &d, double Delta);
    static Eigen::VectorXd jacobiSolve_(const Eigen::MatrixXd &H, const Eigen::VectorXd &r, double shift = 1e-8);


private:
    elasticRod *rod = nullptr;
    timeStepper *stepper = nullptr;
    elasticStretchingForce *m_stretchForce = nullptr;
    elasticBendingForce *m_bendingForce = nullptr;
    externalGravityForce *m_gravityForce = nullptr;




    int freeDOF = 0;

    // reduced workspace
    Eigen::VectorXd force;     // size: freeDOF (∇E on free DOFs) — fill in assembleAt
    Eigen::MatrixXd jacobian;  // size: freeDOF x freeDOF (H on free DOFs)
    Eigen::VectorXd dx;

    using SpMat = Eigen::SparseMatrix<double, Eigen::ColMajor>;
    SpMat JacobianSparse;
};