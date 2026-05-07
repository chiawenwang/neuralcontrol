cmake_minimum_required(VERSION 3.11)
project(py_neural_controller)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
option(USE_MKL "Build with Intel oneMKL/PARDISO" OFF)


set(CMAKE_CXX_FLAGS "-O3 -DNDEBUG -Wno-deprecated-declarations")
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if (COMPILER_SUPPORTS_MARCH_NATIVE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif ()
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "../")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
set(THREADS_PREFER_PTHREAD_FLAG ON)

if (USE_MKL)
    set(MKL_INTERFACE_FULL intel_lp64)
    add_definitions(-DUSE_MKL -DEIGEN_USE_MKL_ALL)
endif ()

find_package(LAPACK REQUIRED)
find_package(BLAS REQUIRED)
find_package(Threads REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenMP)
find_package(pybind11 REQUIRED)
if (USE_MKL)
    find_package(MKL REQUIRED)
endif ()

set(SOURCES
        src/world.cpp
        src/dampingForce.cpp
        src/elasticBendingForce.cpp
        src/elasticStretchingForce.cpp
        src/elasticTwistingForce.cpp
        src/externalGravityForce.cpp
        src/inertialForce.cpp
        src/setInput.cpp
        src/timeStepper.cpp
        src/elasticRod.cpp
        src/solver.cpp
        src/bendingCompute.cpp
)

set(LINK_LIBRARIES
        Eigen3::Eigen
        ${LAPACK_LIBRARIES}
        ${BLAS_LIBRARIES}
        Threads::Threads
        m
        pybind11::module
)
if (USE_MKL)
    set(LINK_LIBRARIES
        MKL::MKL
        ${LINK_LIBRARIES}
    )
endif ()

if (OpenMP_CXX_FOUND)
    set (LINK_LIBRARIES ${LINK_LIBRARIES}
            OpenMP::OpenMP_CXX
    )
endif ()

add_library(common_sources STATIC ${SOURCES})

target_include_directories(common_sources PUBLIC
        ${CMAKE_SOURCE_DIR}/src
)

target_include_directories(common_sources SYSTEM PRIVATE
        Eigen3::Eigen
        )

target_link_libraries(common_sources PUBLIC
        ${LINK_LIBRARIES}
)

pybind11_add_module(nn_der
        src/app.cpp
)
target_link_libraries(nn_der PRIVATE
        common_sources
)

set_target_properties(nn_der PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/nn_der"
)
