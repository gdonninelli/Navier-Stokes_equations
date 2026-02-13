#ifndef NAVIER_STOKES_HPP
#define NAVIER_STOKES_HPP

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria.h>

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
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_description.h>

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>

using namespace dealii;

// Inlet velocity profile, we can change this for different test case
template <unsigned int dim>
class InletVelocity : public Function<dim>
{
public:
  InletVelocity(const double H_       = 0.41,
                const double U_m_     = 1.5,
                const bool   time_dep = true)
    : Function<dim>(dim + 1)
    , H(H_)
    , U_m(U_m_)
    , time_dependent(time_dep)
  {}

  virtual double
  value(const Point<dim> &p, const unsigned int component) const override
  {
    constexpr unsigned int flow_component = (dim == 2) ? 0 : 2;

    if (component == flow_component)
      {
        double profile = 0.0;

        if constexpr (dim == 2)
          {
            const double y = p[1];
            profile = 6.0 * U_m * y * (H - y) / (H * H);
          }
        else if constexpr (dim == 3)
          {
            const double x = p[0];
            const double y = p[1];
            profile =
              16.0 * U_m * x * y * (H - x) * (H - y) / (H * H * H * H);
          }

        if (time_dependent)
          {
            const double t = this->get_time();
            profile *= std::sin(M_PI * t / 8.0);
          }

        return profile;
      }

    return 0.0;
  }

  virtual void
  vector_value(const Point<dim> &p, Vector<double> &values) const override
  {
    for (unsigned int c = 0; c < dim + 1; ++c)
      values[c] = value(p, c);
  }

protected:
  const double H;
  const double U_m;
  const bool   time_dependent;
};

// Homogeneous Dirichlet BC (default, can be overridden for nonzero BC)
template <unsigned int dim>
class ZeroDirichletBC : public Function<dim>
{
public:
  ZeroDirichletBC()
    : Function<dim>(dim + 1)
  {}

  virtual double
  value(const Point<dim> & /*p*/,
        const unsigned int /*component*/) const override
  {
    return 0.0;
  }

  virtual void
  vector_value(const Point<dim> &p, Vector<double> &values) const override
  {
    for (unsigned int c = 0; c < dim + 1; ++c)
      values[c] = value(p, c);
  }
};

// Forcing term f, we can override this if we want to add a source term
template <unsigned int dim>
class ForcingTerm : public Function<dim>
{
public:
  ForcingTerm()
    : Function<dim>(dim + 1)
  {}

  virtual double
  value(const Point<dim> & /*p*/,
        const unsigned int /*component*/) const override
  {
    return 0.0;
  }

  virtual void
  vector_value(const Point<dim> &p, Vector<double> &values) const override
  {
    for (unsigned int c = 0; c < dim + 1; ++c)
      values[c] = value(p, c);
  }
};

// Initial condition, we can override this for nonzero initial conditions
template <unsigned int dim>
class InitialCondition : public Function<dim>
{
public:
  InitialCondition()
    : Function<dim>(dim + 1)
  {}

  virtual double
  value(const Point<dim> & /*p*/,
        const unsigned int /*component*/) const override
  {
    return 0.0;
  }

  virtual void
  vector_value(const Point<dim> &p, Vector<double> &values) const override
  {
    for (unsigned int c = 0; c < dim + 1; ++c)
      values[c] = value(p, c);
  }
};


template <unsigned int dim>
class NavierStokes
{
public:

  // Identity Preconditioner for now
  class PreconditionIdentity
  {
  public:
    void
    vmult(TrilinosWrappers::MPI::BlockVector       &dst,
          const TrilinosWrappers::MPI::BlockVector &src) const
    {
      dst = src;
    }
  };

  // Block-diagonal preconditioner
  class PreconditionBlockDiagonal
  {
  public:
    void
    initialize(const TrilinosWrappers::SparseMatrix &velocity_stiffness_,
               const TrilinosWrappers::SparseMatrix &pressure_mass_)
    {
      velocity_stiffness = &velocity_stiffness_;
      pressure_mass      = &pressure_mass_;

      TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
      amg_data.elliptic              = false;
      amg_data.n_cycles              = 1;
      amg_data.smoother_sweeps       = 2;
      preconditioner_velocity.initialize(velocity_stiffness_, amg_data);
      preconditioner_pressure.initialize(pressure_mass_);
    }

    void
    vmult(TrilinosWrappers::MPI::BlockVector       &dst,
          const TrilinosWrappers::MPI::BlockVector &src) const
    {
      preconditioner_velocity.vmult(dst.block(0), src.block(0));
      preconditioner_pressure.vmult(dst.block(1), src.block(1));
    }

  protected:
    const TrilinosWrappers::SparseMatrix *velocity_stiffness;
    TrilinosWrappers::PreconditionAMG     preconditioner_velocity;

    const TrilinosWrappers::SparseMatrix *pressure_mass;
    TrilinosWrappers::PreconditionILU     preconditioner_pressure;
  };

  // Block-triangular preconditioner
  //
  // For the saddle-point system  [A  G; D  0] [du; dp] = [f; g],
  // the Schur complement is  S = -D A^{-1} G  (negative semi-definite).
  //
  // The lower block-triangular preconditioner is:
  //   P = [A  0; D  C]   with  C ≈ S
  //
  // For time-dependent NS:  A ≈ (rho/dt)M + nu K + conv,
  // so  S ≈ -(dt/rho) M_p  and  C^{-1} should be  -(rho/dt) M_p^{-1}.
  //
  // We apply ILU(M_p)^{-1} and then multiply by pressure_scaling
  // (which should be set to  -rho/dt  by the caller).
  class PreconditionBlockTriangular
  {
  public:
    void
    initialize(const TrilinosWrappers::SparseMatrix &velocity_stiffness_,
               const TrilinosWrappers::SparseMatrix &pressure_mass_,
               const TrilinosWrappers::SparseMatrix &B_,
               const double                          pressure_scaling_ = 1.0)
    {
      velocity_stiffness = &velocity_stiffness_;
      pressure_mass      = &pressure_mass_;
      B                  = &B_;
      pressure_scaling   = pressure_scaling_;

      TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
      amg_data.elliptic              = false;
      amg_data.n_cycles              = 1;
      amg_data.smoother_sweeps       = 2;
      preconditioner_velocity.initialize(velocity_stiffness_, amg_data);
      preconditioner_pressure.initialize(pressure_mass_);
    }

    void
    vmult(TrilinosWrappers::MPI::BlockVector       &dst,
          const TrilinosWrappers::MPI::BlockVector &src) const
    {
      // Step 1: Approximate velocity solve  x ≈ A^{-1} f
      preconditioner_velocity.vmult(dst.block(0), src.block(0));

      // Step 2: Compute  tmp = g - D x  (= g - B*dst0)
      tmp.reinit(src.block(1));
      B->vmult(tmp, dst.block(0));       // tmp = D x
      tmp.sadd(-1.0, src.block(1));      // tmp = g - D x

      // Step 3: Approximate Schur complement solve
      //   y = C^{-1}(g - Dx) ≈ pressure_scaling * M_p^{-1}(g - Dx)
      preconditioner_pressure.vmult(dst.block(1), tmp);
      dst.block(1) *= pressure_scaling;
    }

  protected:
    const TrilinosWrappers::SparseMatrix *velocity_stiffness;
    TrilinosWrappers::PreconditionAMG     preconditioner_velocity;

    const TrilinosWrappers::SparseMatrix *pressure_mass;
    TrilinosWrappers::PreconditionILU     preconditioner_pressure;

    const TrilinosWrappers::SparseMatrix *B;
    double pressure_scaling = 1.0;

    mutable TrilinosWrappers::MPI::Vector tmp;
  };

  // ==========================================================================
  // Constructor
  // ==========================================================================
  // -----------------------------------------------------------------------
  // Constructor.
  // Functions are injected via shared_ptr for maximum flexibility.
  // If a pointer is nullptr, the default implementation is used.
  // -----------------------------------------------------------------------
  // We can inject the functions via shared_ptr, if a pointer is nullptr, the default implementation that are in this file will be used
  NavierStokes(
    const std::string  &mesh_file_name_,
    const unsigned int &degree_velocity_,
    const unsigned int &degree_pressure_,
    const double        deltat_,
    const double        T_,
    const double        Re_,
    const double        U_m_ = 1.5,
    std::shared_ptr<Function<dim>> inlet_velocity_   = nullptr,
    std::shared_ptr<Function<dim>> dirichlet_bc_     = nullptr,
    std::shared_ptr<Function<dim>> forcing_term_     = nullptr,
    std::shared_ptr<Function<dim>> initial_condition_ = nullptr)
    : mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
    , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
    , pcout(std::cout, mpi_rank == 0)
    , mesh_file_name(mesh_file_name_)
    , degree_velocity(degree_velocity_)
    , degree_pressure(degree_pressure_)
    , mesh(MPI_COMM_WORLD)
    , Re(Re_)
    , U_m(U_m_)
    , deltat(deltat_)
    , T(T_)
    , inlet_velocity(inlet_velocity_
                       ? inlet_velocity_
                       : std::make_shared<InletVelocity<dim>>(H, U_m_, true))
    , dirichlet_bc(dirichlet_bc_
                     ? dirichlet_bc_
                     : std::make_shared<ZeroDirichletBC<dim>>())
    , forcing_term(forcing_term_
                     ? forcing_term_
                     : std::make_shared<ForcingTerm<dim>>())
    , initial_condition(initial_condition_
                          ? initial_condition_
                          : std::make_shared<InitialCondition<dim>>())
  {
    fe = std::make_unique<FESystem<dim>>(FE_SimplexP<dim>(degree_velocity),
                                         dim,
                                         FE_SimplexP<dim>(degree_pressure),
                                         1);
    quadrature      = std::make_unique<QGaussSimplex<dim>>(degree_velocity + 1);
    quadrature_face = std::make_unique<QGaussSimplex<dim - 1>>(degree_velocity + 1);
    mapping         = std::make_unique<MappingFE<dim>>(FE_SimplexP<dim>(1));
  }

  // Some setter for changing the functions at runtime if needed
  void
  set_inlet_velocity(std::shared_ptr<Function<dim>> f)
  {
    inlet_velocity = f;
  }

  void
  set_dirichlet_bc(std::shared_ptr<Function<dim>> f)
  {
    dirichlet_bc = f;
  }

  void
  set_forcing_term(std::shared_ptr<Function<dim>> f)
  {
    forcing_term = f;
  }

  void
  set_initial_condition(std::shared_ptr<Function<dim>> f)
  {
    initial_condition = f;
  }

  // Public interface.
  void
  setup();

  void
  assemble();

  void
  solve();

  void
  output(const unsigned int time_step);

  void
  run();

protected:
  const unsigned int mpi_size;
  const unsigned int mpi_rank;
  ConditionalOStream pcout;

  const std::string  mesh_file_name;
  const unsigned int degree_velocity;
  const unsigned int degree_pressure;

  parallel::fullydistributed::Triangulation<dim> mesh;

  // Reynolds Number and related parameters
  double Re;

  // Kinematic viscosity, which can be computed from Re in setup() or set directly
  double nu = 0.001;

  // Fluid density
  double rho = 1.0;

  // Cylinder diameter
  static constexpr double D = 0.1;

  // Channel height
  static constexpr double H = 0.41;

  // Maximum inlet velocity
  double U_m = 1.5;

  // Time-stepping parameters
  double deltat;
  double T;
  double time = 0.0;

  // Newton solver parameters
  static constexpr unsigned int newton_max_iterations = 50;
  static constexpr double       newton_tolerance      = 1e-8;

  // Boundary IDs
  static constexpr unsigned int inlet_boundary_id  = 101;
  static constexpr unsigned int outlet_boundary_id = 102;
  unsigned int wall_boundary_id     = (dim == 2) ? 103 : 104;
  unsigned int cylinder_boundary_id = (dim == 2) ? 104 : 103;

  // Function instances, injected via constructor or setter
  std::shared_ptr<Function<dim>> inlet_velocity;
  std::shared_ptr<Function<dim>> dirichlet_bc;
  std::shared_ptr<Function<dim>> forcing_term;
  std::shared_ptr<Function<dim>> initial_condition;

  // Lift and drag computation.
  void
  compute_lift_drag(double &drag_coeff, double &lift_coeff) const;

  // Pressure difference computation.
  double
  compute_pressure_difference();

  // Newton system (for the nonlinear convective term).
  void
  assemble_newton_system();

  void
  solve_newton_system();

  // FEM
  std::unique_ptr<FiniteElement<dim>>  fe;
  std::unique_ptr<Quadrature<dim>>     quadrature;
  std::unique_ptr<Quadrature<dim - 1>> quadrature_face;

  // Mapping
  std::unique_ptr<MappingFE<dim>> mapping;

  
  DoFHandler<dim> dof_handler;

  IndexSet              locally_owned_dofs;
  std::vector<IndexSet> block_owned_dofs;
  IndexSet              locally_relevant_dofs;
  std::vector<IndexSet> block_relevant_dofs;

  TrilinosWrappers::BlockSparseMatrix system_matrix;
  TrilinosWrappers::BlockSparseMatrix pressure_mass;

  TrilinosWrappers::MPI::BlockVector system_rhs;
  TrilinosWrappers::MPI::BlockVector solution_owned;
  TrilinosWrappers::MPI::BlockVector solution;
  TrilinosWrappers::MPI::BlockVector solution_old;

  TrilinosWrappers::MPI::BlockVector newton_update;
  TrilinosWrappers::MPI::BlockVector current_solution;
};

#endif
