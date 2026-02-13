#include "NavierStokesCN.hpp"

// ============================================================================
// assemble_system()
// ============================================================================
// Assembles the Crank-Nicolson semi-implicit linear system.
//
// System matrix (LHS):
//   [ (1/Δt)M + (ν/2)K + (1/2)C(u*)    G ]   [ u^{n+1} ]   [ rhs_u ]
//   [ D                                  0 ] × [ p^{n+1} ] = [ 0     ]
//
// Right-hand side (RHS):
//   rhs_u = (1/Δt)M u^n − (ν/2)K u^n − (1/2) C(u^n) u^n + f^{n+½}
//
// where:
//   M  = velocity mass matrix
//   K  = velocity stiffness (diffusion) matrix
//   C(w) = convection matrix with transport velocity w
//   G  = pressure gradient operator  (vel-pres block: −(p, ∇·v))
//   D  = divergence operator          (pres-vel block: −(q, ∇·u))
//   u* = extrapolated transport velocity:
//          2 u^n − u^{n−1}  (2nd order)  or  u^n  (1st step)
// ============================================================================

template <unsigned int dim>
void
NavierStokesCN<dim>::assemble_system()
{
  this->system_matrix  = 0;
  this->system_rhs     = 0;
  this->pressure_mass  = 0;

  const QGaussSimplex<dim> quadrature_formula(this->degree_velocity + 1);

  FEValues<dim> fe_values(*this->mapping,
                          *this->fe,
                          quadrature_formula,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = this->fe->dofs_per_cell;
  const unsigned int n_q_points    = quadrature_formula.size();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_pressure_mass(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  // Extractors
  const FEValuesExtractors::Vector velocities(0);
  const FEValuesExtractors::Scalar pressure(dim);

  // Values at quadrature points from old time-step solutions
  std::vector<Tensor<1, dim>> old_velocity_values(n_q_points);
  std::vector<Tensor<2, dim>> old_velocity_gradients(n_q_points);
  std::vector<Tensor<1, dim>> old_old_velocity_values(n_q_points);

  // Basis function values (precomputed per quadrature point)
  std::vector<Tensor<1, dim>> phi_u(dofs_per_cell);
  std::vector<Tensor<2, dim>> grad_phi_u(dofs_per_cell);
  std::vector<double>         phi_p(dofs_per_cell);

  // ---- Boundary conditions at time t^{n+1} (NON-HOMOGENEOUS) ----
  // Unlike the Newton code (which solves for the increment with zero BC),
  // here we solve for u^{n+1} directly with the actual Dirichlet values.
  const ComponentMask velocity_mask = this->fe->component_mask(velocities);

  AffineConstraints<double> constraints;
  constraints.clear();
  constraints.reinit(this->locally_relevant_dofs);

  // Inlet: parabolic profile at t^{n+1}
  VectorTools::interpolate_boundary_values(*this->mapping,
                                           this->dof_handler,
                                           this->inlet_boundary_id,
                                           *this->inlet_velocity,
                                           constraints,
                                           velocity_mask);
  // Walls: no-slip
  VectorTools::interpolate_boundary_values(*this->mapping,
                                           this->dof_handler,
                                           this->wall_boundary_id,
                                           Functions::ZeroFunction<dim>(dim + 1),
                                           constraints,
                                           velocity_mask);
  // Cylinder: no-slip
  VectorTools::interpolate_boundary_values(*this->mapping,
                                           this->dof_handler,
                                           this->cylinder_boundary_id,
                                           Functions::ZeroFunction<dim>(dim + 1),
                                           constraints,
                                           velocity_mask);
  constraints.close();

  // ---- Cell loop ----
  for (const auto &cell : this->dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;

      cell_matrix        = 0;
      cell_rhs           = 0;
      cell_pressure_mass = 0;

      fe_values.reinit(cell);

      // Evaluate previous-time-step fields at quadrature points
      fe_values[velocities].get_function_values(this->solution_old,
                                                old_velocity_values);
      fe_values[velocities].get_function_gradients(this->solution_old,
                                                   old_velocity_gradients);
      fe_values[velocities].get_function_values(solution_old_old,
                                                old_old_velocity_values);

      for (unsigned int q = 0; q < n_q_points; ++q)
        {
          const double JxW = fe_values.JxW(q);

          // Extrapolated transport velocity for the implicit convection
          Tensor<1, dim> u_star;
          if (first_step)
            u_star = old_velocity_values[q]; // 1st-order: u* = u^n
          else
            u_star = 2.0 * old_velocity_values[q]
                     - old_old_velocity_values[q]; // 2nd-order

          // Precompute basis functions
          for (unsigned int k = 0; k < dofs_per_cell; ++k)
            {
              phi_u[k]      = fe_values[velocities].value(k, q);
              grad_phi_u[k] = fe_values[velocities].gradient(k, q);
              phi_p[k]      = fe_values[pressure].value(k, q);
            }

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              // ----- RHS: explicit contributions from time n -----
              // (1/Δt)(u^n · φ_u)
              const double rhs_mass =
                (1.0 / this->deltat) *
                (old_velocity_values[q] * phi_u[i]);

              // −(ν/2)(∇u^n : ∇φ_u)
              const double rhs_visc =
                -(this->nu / 2.0) *
                scalar_product(old_velocity_gradients[q], grad_phi_u[i]);

              // −(1/2)((u^n · ∇)u^n · φ_u)
              const double rhs_conv =
                -0.5 *
                ((old_velocity_values[q] * old_velocity_gradients[q]) *
                 phi_u[i]);

              cell_rhs(i) += (rhs_mass + rhs_visc + rhs_conv) * JxW;

              // ----- LHS: matrix entries -----
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  // 1. Mass from time derivative: (1/Δt)(φ_u^j · φ_u^i)
                  double val =
                    (1.0 / this->deltat) * (phi_u[i] * phi_u[j]);

                  // 2. Viscosity (implicit half):
                  //    (ν/2)(∇φ_u^j : ∇φ_u^i)
                  val += (this->nu / 2.0) *
                         scalar_product(grad_phi_u[i], grad_phi_u[j]);

                  // 3. Semi-implicit convection (implicit half):
                  //    (1/2)(u* · ∇φ_u^j) · φ_u^i
                  val += 0.5 * (u_star * grad_phi_u[j]) * phi_u[i];

                  // 4. Pressure gradient: −(φ_p^j , ∇·φ_u^i)
                  val -= phi_p[j] *
                         fe_values[velocities].divergence(i, q);

                  // 5. Divergence constraint: −(φ_p^i , ∇·φ_u^j)
                  val -= phi_p[i] *
                         fe_values[velocities].divergence(j, q);

                  cell_matrix(i, j) += val * JxW;

                  // Pressure mass matrix (for preconditioner)
                  cell_pressure_mass(i, j) +=
                    phi_p[i] * phi_p[j] * JxW;
                }
            }
        }

      cell->get_dof_indices(local_dof_indices);

      // distribute_local_to_global handles non-homogeneous constraints:
      // it modifies both matrix and RHS so that constrained (Dirichlet)
      // DOFs are set to the prescribed boundary values.
      constraints.distribute_local_to_global(cell_matrix,
                                             cell_rhs,
                                             local_dof_indices,
                                             this->system_matrix,
                                             this->system_rhs);
      constraints.distribute_local_to_global(cell_pressure_mass,
                                             local_dof_indices,
                                             this->pressure_mass);
    }

  this->system_matrix.compress(VectorOperation::add);
  this->system_rhs.compress(VectorOperation::add);
  this->pressure_mass.compress(VectorOperation::add);
}


// ============================================================================
// solve_system()
// ============================================================================
// Solve the assembled linear system with GMRES + block-triangular
// preconditioner (same structure as the Newton code).
//
// The pressure Schur complement approximation is unchanged:
//   S ≈ −(Δt/ρ) M_p     ⟹     S^{−1} ≈ −(ρ/Δt) M_p^{−1}
// ============================================================================

template <unsigned int dim>
void
NavierStokesCN<dim>::solve_system()
{
  const double rhs_norm = this->system_rhs.l2_norm();
  SolverControl solver_control(20000, 1e-6 * rhs_norm);

  // Pressure Schur complement scaling
  const double pressure_scaling = -this->rho / this->deltat;

  typename NavierStokes<dim>::PreconditionBlockTriangular preconditioner;
  preconditioner.initialize(this->system_matrix.block(0, 0),
                            this->pressure_mass.block(1, 1),
                            this->system_matrix.block(1, 0),
                            pressure_scaling);

  // GMRES with restart length 150
  typename SolverGMRES<TrilinosWrappers::MPI::BlockVector>::AdditionalData
    gmres_data(150);
  SolverGMRES<TrilinosWrappers::MPI::BlockVector> solver(solver_control,
                                                          gmres_data);

  this->solution_owned = 0.0;
  solver.solve(this->system_matrix,
               this->solution_owned,
               this->system_rhs,
               preconditioner);

  this->pcout << "  GMRES: " << solver_control.last_step()
              << " iterations" << std::endl;
}


// ============================================================================
// run()
// ============================================================================
// Time-stepping loop with Crank-Nicolson semi-implicit scheme.
//
// Compared to the base-class run() with Newton:
//   - NO Newton iteration: only ONE linear solve per time step
//   - The convection is linearised via 2nd-order extrapolation of the
//     transport velocity
//   - All other aspects (BC, output, lift/drag) are identical
// ============================================================================

template <unsigned int dim>
void
NavierStokesCN<dim>::run()
{
  // 1. Setup (mesh, DOFs, sparsity patterns, vectors)
  this->setup();

  // Allocate the extra vector for u^{n-1}
  solution_old_old.reinit(this->block_owned_dofs,
                          this->block_relevant_dofs,
                          MPI_COMM_WORLD);

  // 2. Initial condition (start from rest)
  this->inlet_velocity->set_time(0.0);

  VectorTools::interpolate(*this->mapping,
                           this->dof_handler,
                           Functions::ZeroFunction<dim>(dim + 1),
                           this->solution_owned);
  this->solution     = this->solution_owned;
  this->solution_old = this->solution;
  solution_old_old   = this->solution;
  this->current_solution = this->solution;

  first_step = true;

  double       time      = 0.0;
  unsigned int time_step = 0;

  // File for benchmark quantities
  std::ofstream forces_file;
  if (this->mpi_rank == 0)
    {
      forces_file.open("forces_cn.txt");
      forces_file << "Time\tCd\tCl\tDeltaP" << std::endl;
    }

  // ---- Time loop ----
  while (time < this->T)
    {
      time += this->deltat;
      time_step++;

      this->pcout << "-----------------------------------------------"
                  << std::endl;
      this->pcout << "Time step " << time_step << "  t = " << time
                  << std::endl;

      // Set BC to time t^{n+1}
      this->inlet_velocity->set_time(time);

      // Assemble the CN linear system
      assemble_system();

      // Solve (single linear solve, no Newton!)
      solve_system();

      // Update ghosted solution
      this->current_solution = this->solution_owned;

      // Shift time levels: u^{n-1} ← u^n,  u^n ← u^{n+1}
      solution_old_old   = this->solution_old;
      this->solution_old = this->current_solution;

      // After the first step, switch to 2nd-order extrapolation
      first_step = false;

      // Compute benchmark quantities
      double drag = 0.0, lift = 0.0;
      this->compute_lift_drag(drag, lift);
      const double delta_p = this->compute_pressure_difference();

      this->pcout << "  Cd = " << drag
                  << "  Cl = " << lift
                  << "  ΔP = " << delta_p << std::endl;

      if (this->mpi_rank == 0)
        {
          forces_file << time << "\t" << drag << "\t" << lift << "\t"
                      << delta_p << std::endl;
          forces_file.flush();
        }

      // VTU output
      this->output(time_step);
    }

  this->pcout << "==============================================="
              << std::endl;
  this->pcout << "Simulation complete." << std::endl;
}

// Explicit template instantiation
template class NavierStokesCN<2>;
template class NavierStokesCN<3>;
