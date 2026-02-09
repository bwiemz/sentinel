cmake_minimum_required(VERSION 3.15)
project(sentinel_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find Python and pybind11
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

# Eigen (header-only) — fetch if not found
find_package(Eigen3 3.4 QUIET)
if(NOT Eigen3_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        Eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
        GIT_SHALLOW TRUE
    )
    set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(Eigen)
endif()

# Core static library
add_library(sentinel_core_lib STATIC
    sentinel_core/src/kalman_ops.cpp
    sentinel_core/src/ekf_ops.cpp
    sentinel_core/src/cost_matrix.cpp
    sentinel_core/src/jpda_ops.cpp
    sentinel_core/src/physics_ops.cpp
    sentinel_core/src/geodetic_ops.cpp
)
target_include_directories(sentinel_core_lib PUBLIC
    sentinel_core/include
)
target_link_libraries(sentinel_core_lib PUBLIC Eigen3::Eigen)

# Platform-specific optimizations
if(MSVC)
    target_compile_options(sentinel_core_lib PRIVATE /O2 /GL)
else()
    target_compile_options(sentinel_core_lib PRIVATE -O3 -march=native -ffast-math)
endif()

# pybind11 Python module
pybind11_add_module(_sentinel_core
    sentinel_core/bindings/module.cpp
    sentinel_core/bindings/bind_kalman.cpp
    sentinel_core/bindings/bind_ekf.cpp
    sentinel_core/bindings/bind_cost_matrix.cpp
    sentinel_core/bindings/bind_jpda.cpp
    sentinel_core/bindings/bind_physics.cpp
    sentinel_core/bindings/bind_geodetic.cpp
)
target_link_libraries(_sentinel_core PRIVATE sentinel_core_lib)

# Install into the sentinel package
install(TARGETS _sentinel_core DESTINATION sentinel)
