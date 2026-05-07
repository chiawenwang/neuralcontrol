#ifndef BENDING_COMPUTE_H
#define BENDING_COMPUTE_H

#include "eigenIncludes.h"
#include "elasticRod.h"
#include "timeStepper.h"
#include <optional>


class bendingCompute
{
public:
	bendingCompute(elasticRod &m_rod, timeStepper &m_stepper);
	~bendingCompute();
    // If coeff is nullptr, EIMat (the internal stiffness) will be used.
    VectorXd computeGrad(MatrixXd kappaBar, const Matrix2d *coeff = nullptr);
    double computeLoss(MatrixXd kappaBar, const Matrix2d *coeff = nullptr);

    double computeStretchLoss(std::optional<double> coeff = std::nullopt);
    VectorXd computeStretchGrad(std::optional<double> coeff = std::nullopt);


private:

	elasticRod *rod;
	timeStepper *stepper;

    int ci;
    double chi;
    double kappa1,kappa2;
    double norm_e,norm_f;
    
    Vector3d te,tf;
    Vector3d d1e,d1f,d2e,d2f;
    Vector3d tilde_t,tilde_d1,tilde_d2;
    Vector3d Dkappa1De,Dkappa1Df,Dkappa2De,Dkappa2Df;
    Vector3d kbLocal;
    Vector2d kappaL;
    VectorXd f;

    MatrixXd gradKappa1;
    MatrixXd gradKappa2;
    MatrixXd relevantPart;

    Matrix2d EIMat;
};

#endif
