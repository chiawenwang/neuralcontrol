#ifndef INERTIALFORCE_H
#define INERTIALFORCE_H

#include "eigenIncludes.h"
#include "elasticRod.h"
#include "timeStepper.h"

class inertialForce
{
public:
	inertialForce(elasticRod &m_rod, timeStepper &m_stepper);
	~inertialForce() = default;
	void computeFi();
	void computeJi();
	void updateEpsilon(double m_epsilon);

private:
	elasticRod *rod;
	timeStepper *stepper;

	double epsilon;
	
    int ind1, ind2, mappedInd1, mappedInd2;	
    double f, jac;
};

#endif
