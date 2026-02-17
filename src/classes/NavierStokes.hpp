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

#include <deal.II/lac/sparsity_tools.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <cmath>
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <iomanip>

using namespace dealii;

enum class TimeScheme     { BackwardEuler, CrankNicolson };
enum class NonlinearMethod { Newton, Linearized };

// Utility: string conversion for printing
inline std::string to_string(TimeScheme s)
{
  return s == TimeScheme::BackwardEuler ? "Backward Euler" : "Crank-Nicolson";
}
inline std::string to_string(NonlinearMethod m)
{
  return m == NonlinearMethod::Newton ? "Newton" : "Linearized (semi-implicit)";
}

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


// =========================================================================
// Configuration struct for benchmark test cases.
// Holds all parameters needed to construct a NavierStokes solver.
// Populated by factory functions in TestCases.hpp.
// =========================================================================
template <unsigned int dim>
struct BenchmarkTestCase
{
  std::string    name;
  std::string    description;
  std::string    mesh_file;
  unsigned int   degree_velocity = 2;
  unsigned int   degree_pressure = 1;
  double         Re;
  double         U_m;
  double         T;
  double         deltat;
  TimeScheme     time_scheme;
  NonlinearMethod nonlinear_method;
  std::shared_ptr<Function<dim>> inlet_velocity;
  std::shared_ptr<Function<dim>> dirichlet_bc;
  std::shared_ptr<Function<dim>> forcing_term;
  std::shared_ptr<Function<dim>> initial_condition;
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
      amg_data.smoother_sweeps       = 3;
      amg_data.aggregation_threshold = 0.02;
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
  // For time-dependent NS:  A ≈ (rho/dt)M + theta*nu*K + conv.
  //
  // Cahouet-Chabard Schur complement approximation:
  //   S^{-1}  ≈  -(rho/dt) K_p^{-1}  -  theta*nu * M_p^{-1}
  //
  // where K_p is the pressure Laplacian (stiffness) and M_p is the
  // pressure mass.  The first term captures the mass-dominated (low
  // frequency) regime and the second the viscosity-dominated (high
  // frequency) regime, giving mesh-independent GMRES iterations.
  class PreconditionBlockTriangular
  {
  public:
    void
    initialize(const TrilinosWrappers::SparseMatrix &velocity_stiffness_,
               const TrilinosWrappers::SparseMatrix &pressure_mass_,
               const TrilinosWrappers::SparseMatrix &pressure_stiffness_,
               const TrilinosWrappers::SparseMatrix &B_,
               const double nu_,
               const double rho_,
               const double deltat_,
               const double theta_)
    {
      velocity_stiffness = &velocity_stiffness_;
      pressure_mass      = &pressure_mass_;
      pressure_stiffness = &pressure_stiffness_;
      B                  = &B_;
      nu_val             = nu_;
      rho_val            = rho_;
      deltat_val         = deltat_;
      theta_val          = theta_;

      // ILU for velocity block (more robust for convection-dominated problems)
      TrilinosWrappers::PreconditionILU::AdditionalData ilu_data_vel;
      ilu_data_vel.ilu_fill = 1; // You can increase for more robustness
      preconditioner_velocity.initialize(velocity_stiffness_, ilu_data_vel);

      // ILU for pressure mass M_p
      preconditioner_pressure_mass.initialize(pressure_mass_);

      // AMG for pressure Laplacian K_p (elliptic operator)
      TrilinosWrappers::PreconditionAMG::AdditionalData amg_data_pres;
      amg_data_pres.elliptic              = true;
      amg_data_pres.n_cycles              = 1;
      amg_data_pres.smoother_sweeps       = 2;
      amg_data_pres.aggregation_threshold = 0.02;
      preconditioner_pressure_lapl.initialize(pressure_stiffness_, amg_data_pres);

      tmp_initialized = false;
    }

    void
    vmult(TrilinosWrappers::MPI::BlockVector       &dst,
          const TrilinosWrappers::MPI::BlockVector &src) const
    {
      // Step 1: Approximate velocity solve  x ≈ A^{-1} f
      preconditioner_velocity.vmult(dst.block(0), src.block(0));

      // Step 2: Compute  tmp = g - D x  (= g - B*dst0)
      // Lazy-init tmp from src to guarantee Epetra_Map compatibility
      if (!tmp_initialized)
        {
          tmp.reinit(src.block(1));
          tmp2.reinit(src.block(1));
          tmp_initialized = true;
        }
      B->vmult(tmp, dst.block(0));       // tmp = D x
      tmp.sadd(-1.0, src.block(1));      // tmp = g - D x

      // Step 3: Cahouet-Chabard Schur complement solve
      //   S^{-1} ≈ -(rho/dt)*K_p^{-1} - theta*nu*M_p^{-1}
      //
      // Term 1 (dominant): -(rho/dt) * K_p^{-1} * (g - Dx)
      preconditioner_pressure_lapl.vmult(dst.block(1), tmp);
      dst.block(1) *= -(rho_val / deltat_val);

      // Term 2: -theta*nu * M_p^{-1} * (g - Dx)
      preconditioner_pressure_mass.vmult(tmp2, tmp);
      dst.block(1).add(-(theta_val * nu_val), tmp2);
    }

  protected:
    const TrilinosWrappers::SparseMatrix *velocity_stiffness;
    TrilinosWrappers::PreconditionILU     preconditioner_velocity;

    const TrilinosWrappers::SparseMatrix *pressure_mass;
    TrilinosWrappers::PreconditionILU     preconditioner_pressure_mass;

    const TrilinosWrappers::SparseMatrix *pressure_stiffness;
    TrilinosWrappers::PreconditionAMG     preconditioner_pressure_lapl;

    const TrilinosWrappers::SparseMatrix *B;

    double nu_val     = 0.001;
    double rho_val    = 1.0;
    double deltat_val = 0.02;
    double theta_val  = 1.0;

    mutable TrilinosWrappers::MPI::Vector tmp;
    mutable TrilinosWrappers::MPI::Vector tmp2;
    mutable bool tmp_initialized = false;
  };

  // ==========================================================================
  // Constructor
  // ==========================================================================
  // -----------------------------------------------------------------------
  // Constructor.
  // Functions are injected via shared_ptr for maximum flexibility.
  // If a pointer is nullptr, the default implementation is used.
  // -----------------------------------------------------------------------
  // Static helper: suggested dt based on Reynolds number
  // -----------------------------------------------------------------------
  static double compute_default_deltat(double Re)
  {
    if (Re <= 20)       return 0.1;
    else if (Re <= 50)  return 0.05;
    else if (Re <= 100) return 0.02;
    else if (Re <= 150) return 0.01;
    else                return 0.005;
  }

  // -----------------------------------------------------------------------
  // Constructor from BenchmarkTestCase (delegates to full constructor).
  // Usage:  auto tc = TestCases::make_2D_3(mesh_file);
  //         NavierStokes<2> solver(tc);
  // -----------------------------------------------------------------------
  NavierStokes(const BenchmarkTestCase<dim> &tc)
    : NavierStokes(tc.mesh_file, tc.degree_velocity, tc.degree_pressure,
                   tc.deltat, tc.T, tc.Re, tc.U_m,
                   tc.time_scheme, tc.nonlinear_method,
                   tc.inlet_velocity, tc.dirichlet_bc,
                   tc.forcing_term, tc.initial_condition)
  {}

  // -----------------------------------------------------------------------
  // Constructor.
  // deltat_ <= 0 means auto-select based on Re.
  // -----------------------------------------------------------------------
  NavierStokes(
    const std::string  &mesh_file_name_,
    const unsigned int &degree_velocity_,
    const unsigned int &degree_pressure_,
    const double        deltat_,
    const double        T_,
    const double        Re_,
    const double        U_m_ = 1.5,
    TimeScheme          time_scheme_      = TimeScheme::BackwardEuler,
    NonlinearMethod     nonlinear_method_ = NonlinearMethod::Newton,
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
    , deltat(deltat_ > 0 ? deltat_ : compute_default_deltat(Re_))
    , T(T_)
    , time_scheme(time_scheme_)
    , nonlinear_method(nonlinear_method_)
    , theta(time_scheme_ == TimeScheme::CrankNicolson ? 0.5 : 1.0)
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

  // Method selection
  TimeScheme      time_scheme;
  NonlinearMethod nonlinear_method;
  double          theta;  // 1.0 for BE, 0.5 for CN

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

  // Newton system: assembles Jacobian + residual using theta-method.
  void
  assemble_newton_system();

  // Solve Newton linear system for delta u.
  void
  solve_newton_system();

  // Linearized (semi-implicit) system: one linear solve per step.
  // Convection linearized via extrapolation of transport velocity.
  void
  assemble_linearized_system();

  // Solve the linearized system for u^{n+1} directly.
  // Returns true if converged, false otherwise (does not throw on failure).
  bool
  solve_linear_system();

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
  TrilinosWrappers::BlockSparseMatrix pressure_stiffness; // Pressure Laplacian K_p

  TrilinosWrappers::MPI::BlockVector system_rhs;
  TrilinosWrappers::MPI::BlockVector solution_owned;
  TrilinosWrappers::MPI::BlockVector solution;
  TrilinosWrappers::MPI::BlockVector solution_old;

  TrilinosWrappers::MPI::BlockVector newton_update;
  TrilinosWrappers::MPI::BlockVector current_solution;

  // Backup vector for Newton damping (avoids reallocation every time step)
  TrilinosWrappers::MPI::BlockVector solution_backup;

  // For CN: u^{n-1} (needed for 2nd-order extrapolation of transport velocity)
  TrilinosWrappers::MPI::BlockVector solution_old_old;
  bool first_step = true;

  // Flag: pressure mass matrix only needs to be assembled once (mesh-dependent only)
  bool pressure_mass_assembled = false;

  // Flag: pressure stiffness (Laplacian) only assembled once
  bool pressure_stiffness_assembled = false;

  // Homogeneous Dirichlet constraints for Newton updates (built once in setup)
  AffineConstraints<double> newton_constraints;

  // Non-homogeneous Dirichlet constraints for linearized approach (rebuilt each step)
  AffineConstraints<double> system_constraints;

  // PVD record: stores (time, filename) pairs for ParaView collection
  std::vector<std::pair<double, std::string>> pvd_records;

  // Write PVD file collecting all time steps
  void write_pvd_file() const;
};

#endif
