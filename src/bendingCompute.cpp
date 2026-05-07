#include "bendingCompute.h"

bendingCompute::bendingCompute(elasticRod &m_rod, timeStepper &m_stepper)
{
	rod = &m_rod;
	stepper = &m_stepper;

	double EI = rod->EI;
    EIMat<<EI,0,
           0,EI;

	int nv = rod->nv;
    gradKappa1 = MatrixXd::Zero(nv,11);
    gradKappa2 = MatrixXd::Zero(nv,11);
    relevantPart = MatrixXd::Zero(11, 2);
    f = VectorXd::Zero(11);
}

bendingCompute::~bendingCompute()
{
	;
}

double bendingCompute::computeLoss(MatrixXd kappaBar, const Matrix2d *coeff)
{
    double loss = 0.0;
    for (int i=1; i < rod->ne; i++)
    {
        kappaL = (rod->kappa).row(i) - kappaBar.row(i);
        const Matrix2d &stiff = (coeff == nullptr) ? EIMat : *coeff;

        double E_b = 0.5 * (kappaL.transpose() * (stiff * kappaL)).value() / rod->voronoiLen(i);
        loss += E_b;    
    }
    return loss;
}

double bendingCompute::computeStretchLoss(std::optional<double> coeff)
{
    const double c = coeff.value_or(rod->EA);

    double stretchLoss = 0.0;
    for (int i=0;i<rod->ne;i++)
    {
        double epsX = rod->edgeLen(i) / rod->refLen(i) - 1.0;
        stretchLoss += 0.5 * c * epsX * epsX * rod->refLen(i);

        // cout << "i: " << i << ", epsX: " << epsX << ", stretchLoss: " << stretchLoss << endl;
    }
    return stretchLoss;
}

VectorXd bendingCompute::computeStretchGrad(std::optional<double> coeff)
{
    const double c = coeff.value_or(rod->EA);

    VectorXd stretchGrad = VectorXd::Zero(rod->ndof);    

    for (int i=0;i<rod->ne;i++)
    {
        double epsX = rod->edgeLen(i) / rod->refLen(i) - 1.0;
        f = c * epsX * (rod->tangent).row(i);

        for (int k = 0; k < 3; k++)
        {
            int ind1 = 4*i + k;
            stretchGrad(ind1)   += f[k];

            int ind2 = 4*(i + 1) + k;
            stretchGrad(ind2)   -= f[k];
        }
    }

    // 4n + 3 to 3 * n
    VectorXd stretchGrad_result = VectorXd::Zero(3 * rod->nv);
    for (int i = 0; i < rod->nv; i++)
    {
        stretchGrad_result(3 * i) = stretchGrad(4 * i);
        stretchGrad_result(3 * i + 1) = stretchGrad(4 * i + 1);
        stretchGrad_result(3 * i + 2) = stretchGrad(4 * i + 2);
    }

    return -stretchGrad_result;
}


VectorXd bendingCompute::computeGrad(MatrixXd kappaBar, const Matrix2d *coeff)
{
    // dkap / dx
    VectorXd dkap = VectorXd::Zero(rod->ndof);    

    for(int i=1;i<rod->ne;i++)
    {
        norm_e = rod->edgeLen(i-1);
        norm_f = rod->edgeLen(i);
        te = rod->tangent.row(i-1);
        tf = rod->tangent.row(i);
        d1e = rod->m1.row(i-1);
        d2e = rod->m2.row(i-1);
        d1f = rod->m1.row(i);
        d2f = rod->m2.row(i);

        chi = 1.0 + te.dot(tf);
        tilde_t = (te+tf)/chi;
        tilde_d1 = (d1e+d1f)/chi;
        tilde_d2 = (d2e+d2f)/chi;

        kappa1 = rod->kappa(i,0);
        kappa2 = rod->kappa(i,1);

        Dkappa1De = (1.0/norm_e)*(-kappa1*tilde_t + tf.cross(tilde_d2));
        Dkappa1Df = (1.0/norm_f)*(-kappa1*tilde_t - te.cross(tilde_d2));
        Dkappa2De = (1.0/norm_e)*(-kappa2*tilde_t - tf.cross(tilde_d1));
        Dkappa2Df = (1.0/norm_f)*(-kappa2*tilde_t + te.cross(tilde_d1));

        gradKappa1.row(i).segment(0,3)=-Dkappa1De;
        gradKappa1.row(i).segment(4,3)= Dkappa1De - Dkappa1Df;
        gradKappa1.row(i).segment(8,3)= Dkappa1Df;

        gradKappa2.row(i).segment(0,3)=-Dkappa2De;
        gradKappa2.row(i).segment(4,3)= Dkappa2De - Dkappa2Df;
        gradKappa2.row(i).segment(8,3)= Dkappa2Df;

        kbLocal = (rod->kb).row(i);

        gradKappa1(i,3)=-0.5*kbLocal.dot(d1e);
        gradKappa1(i,7)=-0.5*kbLocal.dot(d1f);
        gradKappa2(i,3)=-0.5*kbLocal.dot(d2e);
        gradKappa2(i,7)=-0.5*kbLocal.dot(d2f);
    }

    for (int i=1; i < rod->ne; i++)
    {
        ci = 4*i-4;
        relevantPart.col(0) = gradKappa1.row(i);
        relevantPart.col(1) = gradKappa2.row(i);
        kappaL = (rod->kappa).row(i) - kappaBar.row(i);
        // choose stiffness matrix: provided coeff or internal EIMat
        const Matrix2d &stiff = (coeff == nullptr) ? EIMat : *coeff;
        f = relevantPart * stiff * kappaL / rod->voronoiLen(i);
        
        for (int k = 0; k < 11; k++)
		{
			int ind = ci + k;
            dkap(ind) += f[k];
		}
    }

    // 4n + 3 to 3 * n
    VectorXd dkap_result = VectorXd::Zero(3 * rod->nv);
    for (int i = 0; i < rod->nv; i++)
    {
        dkap_result(3 * i) = dkap(4 * i);
        dkap_result(3 * i + 1) = dkap(4 * i + 1);
        dkap_result(3 * i + 2) = dkap(4 * i + 2);
    }


    return dkap_result;
}
