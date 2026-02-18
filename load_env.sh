#!/bin/bash

echo "Pulizia dell'ambiente in corso..."
module purge

echo "Caricamento dei moduli base e compilatori..."
module load profile/base
module load gcc/12.2.0

module load spack

# Carico dealii con spack su una variabile d'ambiente
export DEAL_II_DIR=$(spack location -i dealii)
export MPI_DIR=$(spack location -i mpich)
export CMAKE_DIR=$(spack location -i /pjgva3e)
export BOOST_DIR=$(spack location -i /ykgcks7)
export CMAKE_PREFIX_PATH=$BOOST_DIR:$CMAKE_PREFIX_PATH
export GMSH_DIR=$(spack location -i gmsh)

export PATH=$MPI_DIR/bin:$CMAKE_DIR/bin:$GMSH_DIR/bin:$PATH

export CC=$MPI_DIR/bin/mpicc
export CXX=$MPI_DIR/bin/mpicxx

echo "Moduli caricati con successo! Ecco il tuo ambiente attuale:"
module list

echo "Per compilare: cmake -DDEAL_II_DIR=\$DEAL_II_DIR .."
