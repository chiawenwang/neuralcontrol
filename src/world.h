#ifndef WORLD_H
#define WORLD_H

#include <memory>
#include "eigenIncludes.h"

// include elastic rod class
#include "elasticRod.h"

// include force classes
#include "elasticStretchingForce.h"
#include "elasticBendingForce.h"
#include "elasticTwistingForce.h"
#include "externalGravityForce.h"
#include "inertialForce.h"

// include external force
#include "dampingForce.h"

// include time stepper
#include "timeStepper.h"

// include input file and option
#include "setInput.h"

#include "solver.h"
#include "bendingCompute.h"


class world
{
public:
	world() = default;
    world(const world&) = delete;               // no copying
    world& operator=(const world&) = delete;    // no assignment

    world(world&&) = default;                   // allow move
    world& operator=(world&&) = default;        // allow move
	world(setInput &m_inputData);

	
	~world();
	void setRodStepper();
	void updateTimeStep();
	void updateTimeStepWithInertia();

	bool simulationRunning();
	int numPoints();
	double getScaledCoordinate(int i);
	double getCurrentTime();
	double getTotalTime();

	bool isRender();

	// file output
	void OpenFile(ofstream &outfile);
	void CloseFile(ofstream &outfile);
	void CoutData(ofstream &outfile);

	void newtonMethod(bool &solved);
	
	// Physical parameters
	double RodLength;
	double rodRadius;
	int numVertices;
	double youngM;
	double Poisson;
	double shearM;
	double deltaTime;
	double totalTime;
	double density;
	Vector3d gVector;
	double viscosity;
	
	bool render; // should the OpenGL rendering be included?
	bool saveData; // should data be written to a file?

	MatrixXd getAllCoordinates();

	VectorXd getForce();
	MatrixXd getJacobian();
	VectorXd getStretchForce();

	void resetSim();
	void rodBoundaryCondition();

	void defineController(const Eigen::MatrixXi &control_inputs);
	void updateControlInputs(const Eigen::MatrixXd &control_inputs);

	double tol, stol;
	int maxIter; // maximum number of iterations
	string fileName;

	void cleanup();

	double getVelocity();

	MatrixXd computeCurvature(const Eigen::MatrixXd &vertices);

	VectorXd computeDCurvature(const Eigen::MatrixXd &kappa_bar, const Matrix2d *coeff = nullptr);
	double computeCurvatureLoss(const Eigen::MatrixXd &kappa_bar, const Matrix2d *coeff = nullptr);

	VectorXd computeStretchGrad(std::optional<double> coeff = std::nullopt);
	double computeStretchLoss(std::optional<double> coeff = std::nullopt);

	MatrixXd getAllFrames();
	void setAllFrames(const Eigen::MatrixXd &m1);
	void setAllVertices(const Eigen::VectorXd &X);

private:
	double characteristicForce;
	double forceTol;

	// Geometry
	MatrixXd vertices;
	MatrixXd vertices0;
	VectorXd theta;
	double currentTime;

	// shared resources (rod and stepper)
	shared_ptr<elasticRod> rod;
	shared_ptr<timeStepper> stepper;
	shared_ptr<solver> tr_solver;


	// unique resources (forces)
	unique_ptr<elasticStretchingForce> m_stretchForce;
	unique_ptr<elasticBendingForce> m_bendingForce;
	unique_ptr<elasticTwistingForce> m_twistingForce;
	unique_ptr<inertialForce> m_inertialForce;
	unique_ptr<externalGravityForce> m_gravityForce;
	unique_ptr<dampingForce> m_dampingForce;
	unique_ptr<bendingCompute> m_bendingCompute;

	int Nstep;
	int timeStep;
	int iter;

	void rodGeometry();

	MatrixXi control_info;
	MatrixXd control_inputs;

	bool hasConverged(const MatrixXd &current_status);
	void getControlStatus(MatrixXd &current_status);
	void updateControlStatus(const MatrixXd &control_status);

	solver::TROpts opts;


};

#endif
