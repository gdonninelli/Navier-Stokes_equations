#ifndef NAVIER_STOKES_2D_HPP
#define NAVIER_STOKES_2D_HPP

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_extractors.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>

using namespace dealii;

class NavierStokes2D
{
public:
  static constexpr unsigned int dim = 2;

  class PreconditionIdentity
  {
  public:
    void
    vmult(TrilinosWrappers::MPI::BlockVector       &dst,
          const TrilinosWrappers::MPI::BlockVector &src) const
    {
      dst = src;
    }

  protected:
  };

  class PreconditionBlockDiagonal
  {
  public:
    void
    initialize(const TrilinosWrappers::SparseMatrix &velocity_stiffness_,
               const TrilinosWrappers::SparseMatrix &pressure_mass_)
    {
      velocity_stiffness = &velocity_stiffness_;
      pressure_mass      = &pressure_mass_;

      preconditioner_velocity.initialize(velocity_stiffness_);
      preconditioner_pressure.initialize(pressure_mass_);
    }

    void
    vmult(TrilinosWrappers::MPI::BlockVector       &dst,
          const TrilinosWrappers::MPI::BlockVector &src) const
    {
      SolverControl                           solver_control_velocity(1000,
                                            1e-2 * src.block(0).l2_norm());
      SolverGMRES<TrilinosWrappers::MPI::Vector> solver_cg_velocity(
        solver_control_velocity);
      solver_cg_velocity.solve(*velocity_stiffness,
                               dst.block(0),
                               src.block(0),
                               preconditioner_velocity);

      SolverControl                           solver_control_pressure(1000,
                                            1e-2 * src.block(1).l2_norm());
      SolverGMRES<TrilinosWrappers::MPI::Vector> solver_cg_pressure(
        solver_control_pressure);
      solver_cg_pressure.solve(*pressure_mass,
                               dst.block(1),
                               src.block(1),
                               preconditioner_pressure);
    }

  protected:
    const TrilinosWrappers::SparseMatrix *velocity_stiffness;

    TrilinosWrappers::PreconditionILU preconditioner_velocity;

    const TrilinosWrappers::SparseMatrix *pressure_mass;

    TrilinosWrappers::PreconditionILU preconditioner_pressure;
  };

  class PreconditionBlockTriangular
  {
  public:
    void
    initialize(const TrilinosWrappers::SparseMatrix &velocity_stiffness_,
               const TrilinosWrappers::SparseMatrix &pressure_mass_,
               const TrilinosWrappers::SparseMatrix &B_)
    {
      velocity_stiffness = &velocity_stiffness_;
      pressure_mass      = &pressure_mass_;
      B                  = &B_;

      preconditioner_velocity.initialize(velocity_stiffness_);
      preconditioner_pressure.initialize(pressure_mass_);
    }

    // Application of the preconditioner.
    void
    vmult(TrilinosWrappers::MPI::BlockVector       &dst,
          const TrilinosWrappers::MPI::BlockVector &src) const
    {
      SolverControl                           solver_control_velocity(1000,
                                            1e-2 * src.block(0).l2_norm());
      SolverGMRES<TrilinosWrappers::MPI::Vector> solver_cg_velocity(
        solver_control_velocity);
      solver_cg_velocity.solve(*velocity_stiffness,
                               dst.block(0),
                               src.block(0),
                               preconditioner_velocity);

      tmp.reinit(src.block(1));
      B->vmult(tmp, dst.block(0));
      tmp.sadd(-1.0, src.block(1));

      SolverControl                           solver_control_pressure(1000,
                                            1e-2 * src.block(1).l2_norm());
      SolverGMRES<TrilinosWrappers::MPI::Vector> solver_cg_pressure(
        solver_control_pressure);
      solver_cg_pressure.solve(*pressure_mass,
                               dst.block(1),
                               tmp,
                               preconditioner_pressure);
    }

  protected:
    const TrilinosWrappers::SparseMatrix *velocity_stiffness;

    TrilinosWrappers::PreconditionILU preconditioner_velocity;

    const TrilinosWrappers::SparseMatrix *pressure_mass;

    TrilinosWrappers::PreconditionILU preconditioner_pressure;

    const TrilinosWrappers::SparseMatrix *B;

    mutable TrilinosWrappers::MPI::Vector tmp;
  };

  // Constructor.
  NavierStokes2D(const std::string  &mesh_file_name_,
         const unsigned int &degree_velocity_,
         const unsigned int &degree_pressure_,
         double             deltat_,
         double             T_,
         const std::function<double(const Point<dim> &)> &f_)
    : mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
    , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
    , pcout(std::cout, mpi_rank == 0)
    , mesh_file_name(mesh_file_name_)
    , degree_velocity(degree_velocity_)
    , degree_pressure(degree_pressure_)
    , mesh(MPI_COMM_WORLD)
    , deltat(deltat_)
    , T(T_)
    , f(f_)
  {}

  void
  setup();

  void
  assemble();

  void
  solve();

  void
  output(const unsigned int time_step);

  void 
  run(); // TODO: generico, start del ciclo

protected:
  // MPI parallel. /////////////////////////////////////////////////////////////

  const unsigned int mpi_size;

  const unsigned int mpi_rank;

  ConditionalOStream pcout;

  // Problem definition. ///////////////////////////////////////////////////////

  const double nu = 0.001;

  double deltat;

  double T = 8.0; // Like in the beanchMark, but can be changed

  double time = 0.0;

  double rho = 1.0; // Fluid density

  const unsigned int cylinder_boundary_id = 0; // TODO: Identificatore del bordo del cilindro, da verificare con gmsh 

  void assemble_newton_system(); // TODO: per calcolo non lineare

  void solve_newton_system(); // TODO: per calcolo non lineare

   void 
  compute_lift_drag(double &lift_coeff, double &drag_coeff, double time) const; // TODO: helper per tracciare lift e drag

  // Discretization. ///////////////////////////////////////////////////////////

  const std::string mesh_file_name;

  const unsigned int degree_velocity;

  const unsigned int degree_pressure;

  parallel::fullydistributed::Triangulation<dim> mesh;

  std::unique_ptr<FiniteElement<dim>> fe;

  std::unique_ptr<Quadrature<dim>> quadrature;

  std::unique_ptr<Quadrature<dim - 1>> quadrature_face;

  DoFHandler<dim> dof_handler;

  IndexSet locally_owned_dofs;

  std::vector<IndexSet> block_owned_dofs;

  IndexSet locally_relevant_dofs;

  std::vector<IndexSet> block_relevant_dofs;

  TrilinosWrappers::BlockSparseMatrix system_matrix;

  TrilinosWrappers::BlockSparseMatrix pressure_mass;

  TrilinosWrappers::MPI::BlockVector system_rhs;

  TrilinosWrappers::MPI::BlockVector solution_owned;

  TrilinosWrappers::MPI::BlockVector solution;

  TrilinosWrappers::MPI::BlockVector solution_old;

  // Navier Stokes adjustement for non linearity

  TrilinosWrappers::MPI::BlockVector newton_update;

  TrilinosWrappers::MPI::BlockVector current_solution;
};

#endif