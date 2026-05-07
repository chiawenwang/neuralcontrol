#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "world.h"

namespace py = pybind11;


class SimulationManager{
public:
    SimulationManager() {
        w = std::make_shared<world>();
    }
    ~SimulationManager() = default;

    void configure(const pybind11::dict &config){
        if (config.contains("youngM")){
            w->youngM = config["youngM"].cast<double>();
        }
        if (config.contains("Poisson")){
            w->Poisson = config["Poisson"].cast<double>();
        }
        if (config.contains("deltaTime")){
            w->deltaTime = config["deltaTime"].cast<double>();
        }
        if (config.contains("totalTime")){
            w->totalTime = config["totalTime"].cast<double>();
        }
        if (config.contains("gVector")){
            w->gVector = config["gVector"].cast<Eigen::Vector3d>();
        }
        if (config.contains("viscosity")){
            w->viscosity = config["viscosity"].cast<double>();
        }
        if (config.contains("tol")){
            w->tol = config["tol"].cast<double>();
        }
        if (config.contains("stol")){
            w->stol = config["stol"].cast<double>();
        }
        if (config.contains("rodRadius")){
            w->rodRadius = config["rodRadius"].cast<double>();
        }
        if (config.contains("maxIter")){
            w->maxIter = config["maxIter"].cast<int>();
        }
        if (config.contains("density")){
            w->density = config["density"].cast<double>();
        }
        if (config.contains("geometry_file")){
            w->fileName = config["geometry_file"].cast<std::string>();
        }
        initialize();
    } 


    void setControlInputs(const Eigen::MatrixXd &control_inputs) {
        w->updateControlInputs(control_inputs);
    }



    void defineController(const Eigen::MatrixXi &control_inputs){
        w->defineController(control_inputs);
    }


    void initialize() {
        w->setRodStepper();
    }

    void resetSim() {
        w->resetSim();
    }

    double getVelocity() {
        return w->getVelocity();
    }


    bool simulationCompleted() {
        return !w->simulationRunning();
    }
    void step() {
        w->updateTimeStep();
    }
    
    void stepWithInertia() {
        w->updateTimeStepWithInertia();
    }

    Eigen::VectorXd getForce() {
        return w->getForce();
    }
    
    Eigen::MatrixXd getJacobian() {
        return w->getJacobian();
    }

    MatrixXd getAllCoordinates() {
        MatrixXd coordinates = w->getAllCoordinates();
        return coordinates;
    }

    MatrixXd getAllFrames() {
        return w->getAllFrames();
    }

    void setAllFrames(const Eigen::MatrixXd &frames) {
        w->setAllFrames(frames);
    }

    void setAllVertices(const Eigen::VectorXd &X) {
        w->setAllVertices(X);
    }


    void cleanup() {
        if (w){
            w->cleanup();
            w.reset();  // reset the shared pointer to delete the world
        }
    }

    VectorXd getStretchForce(){
        return w->getStretchForce();
    }

    MatrixXd computeCurvature(const MatrixXd &vertices){

        return w->computeCurvature(vertices);
    }

    VectorXd computeDCurvature(const Eigen::MatrixXd &kappa_bar, const Matrix2d *coeff = nullptr){
        return w->computeDCurvature(kappa_bar, coeff);
    }

    double computeCurvatureLoss(const Eigen::MatrixXd &kappa_bar, const Matrix2d *coeff = nullptr){
        return w->computeCurvatureLoss(kappa_bar, coeff);
    }   

    double computeStretchLoss(std::optional<double> coeff = std::nullopt){
        return w->computeStretchLoss(coeff);
    }
    VectorXd computeStretchGrad(std::optional<double> coeff = std::nullopt){
        return w->computeStretchGrad(coeff);
    }



private:
    std::shared_ptr<world> w;


};



PYBIND11_MODULE(nn_der, m){
    m.doc() = "Simulation module for plate dynamics";

        py::class_<SimulationManager>(m, "SimulationManager")
        .def(py::init<>())
        .def("configure", &SimulationManager::configure)
        .def("initialize", &SimulationManager::initialize)
        .def("simulationCompleted", &SimulationManager::simulationCompleted)
        .def("step", &SimulationManager::step)
        .def("getAllVertices", &SimulationManager::getAllCoordinates)
        .def("getForce", &SimulationManager::getForce)
        .def("getJacobian", &SimulationManager::getJacobian)
        .def("resetSim", &SimulationManager::resetSim)
        .def("setControlInputs", &SimulationManager::setControlInputs)
        .def("defineController", &SimulationManager::defineController)
        .def("cleanup", &SimulationManager::cleanup)
        .def("getStretchForce", &SimulationManager::getStretchForce)
        .def("getVelocity", &SimulationManager::getVelocity)
        .def("compute_curvature", &SimulationManager::computeCurvature)
        .def("compute_dcurvature", &SimulationManager::computeDCurvature, py::arg("kappa_bar"), py::arg("coeff") = nullptr)
        .def("compute_curvature_loss", &SimulationManager::computeCurvatureLoss, py::arg("kappa_bar"), py::arg("coeff") = nullptr)
        .def("compute_stretch_loss", &SimulationManager::computeStretchLoss, py::arg("coeff") = std::nullopt)
        .def("compute_stretch_grad", &SimulationManager::computeStretchGrad, py::arg("coeff") = std::nullopt)
        .def("get_all_frames", &SimulationManager::getAllFrames)
        .def("set_all_frames", &SimulationManager::setAllFrames)
        .def("set_all_vertices", &SimulationManager::setAllVertices)
        .def("step_with_inertia", &SimulationManager::stepWithInertia);
}