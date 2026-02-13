cmake_minimum_required(VERSION 3.12.0)
project(NavierStokesSolver LANGUAGES CXX)

# Find deal.II installation
# It is usually found automatically if installed in /usr or /usr/local
find_package(deal.II REQUIRED)

# Add inclusion paths for project headers
include_directories(${CMAKE_SOURCE_DIR}/src/classes)

# Define the source files
set(SOURCE_FILES
    src/main.cc
    src/classes/NavierStokes.cpp
)

# Define the executable
add_executable(navier_stokes ${SOURCE_FILES})

# Link against deal.II and its dependencies (MPI, Trilinos, etc.)
deal_ii_setup_target(navier_stokes)
