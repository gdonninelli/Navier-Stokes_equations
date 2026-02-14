#ifndef NAVIER_STOKES_CN_HPP
#define NAVIER_STOKES_CN_HPP

#include "NavierStokes.hpp"

// ============================================================================
// NavierStokesCN — Semi-implicit Crank-Nicolson solver
// ============================================================================
//
// This class solves the incompressible Navier-Stokes equations using:
//
//   1. Crank-Nicolson (θ = 0.5) time discretization — O(Δt²) accuracy
//   2. Semi-implicit treatment of convection — linearizes the system,
//      so NO Newton iteration is needed (one linear solve per time step)
//
// The semi-discrete formulation is (in weak form):
//
//   (u^{n+1} - u^n)/Δt
//     + (1/2) [(u* · ∇)u^{n+1}  +  (u^n · ∇)u^n]
//     - (ν/2) [Δu^{n+1}  +  Δu^n]
//     + ∇p^{n+1}  =  f^{n+1/2}
//
//   ∇ · u^{n+1}  =  0
//
// where  u* = 2 u^n − u^{n−1}  (2nd-order extrapolation of the transport
//        velocity; falls back to  u* = u^n  at the first time step).
//
// The resulting LINEAR saddle-point system is solved with GMRES +
// block-triangular preconditioner (AMG on velocity, ILU on pressure mass).
//
// All other aspects (mesh, FE spaces, boundary conditions, lift/drag, output)
// are inherited from the base NavierStokes<dim> class.
// ============================================================================

template <unsigned int dim>
class NavierStokesCN : public NavierStokes<dim>
{
public:
  // Inherit the base-class constructor (same parameters)
  using NavierStokes<dim>::NavierStokes;

  // Main entry point (hides the base-class run())
  void
  run();

protected:
  // Assemble the CN semi-implicit linear system
  void
  assemble_system();

  // Solve the assembled linear system
  void
  solve_system();

  // Extra vector for 2nd-order extrapolation: u^{n-1}
  TrilinosWrappers::MPI::BlockVector solution_old_old;

  // Constraints for the current time step (non-homogeneous Dirichlet)
  AffineConstraints<double> cn_constraints;

  // Flag: first time step uses 1st-order extrapolation (u* = u^n)
  bool first_step = true;
};

#endif
