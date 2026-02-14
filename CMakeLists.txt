cmake_minimum_required(VERSION 3.12.0)
project(NavierStokesSolver LANGUAGES CXX C)

include(common/cmake-common.cmake)

add_executable(navier_stokes src/main.cpp src/classes/NavierStokes.cpp)
deal_ii_setup_target(navier_stokes)