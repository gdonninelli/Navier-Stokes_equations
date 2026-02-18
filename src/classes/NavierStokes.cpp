#include "NavierStokes.hpp"

template <unsigned int dim> void NavierStokes<dim>::setup() {
  pcout << "===============================================" << std::endl;
  pcout << "Setup..." << std::endl;

  // TODO: Controllare se va bene, proabilmente mesh non nativamente supprotata
  // da deal.II, quindi dobbiamo leggere in un triangulation seriale e poi
  // distribuirla. 0. Mesh loading For parallel::fullydistributed::Triangulation
  // we must:
  //   a) Read the mesh into a serial Triangulation
  //   b) Partition it
  //   c) Create the distributed triangulation from the description
  {
    Triangulation<dim> serial_mesh;
    GridIn<dim> grid_in;
    grid_in.attach_triangulation(serial_mesh);

    std::ifstream input_file(mesh_file_name);
    AssertThrow(input_file.is_open(),
                ExcMessage("Could not open mesh file: " + mesh_file_name));

    // Read the file, replacing $ParametricNodes with $Nodes
    // because deal.II does not support $ParametricNodes
    std::stringstream ss;
    std::string line;
    bool in_parametric_nodes = false;
    bool first_line_after_header =
        false; // the count line right after $ParametricNodes
    while (std::getline(input_file, line)) {
      // Strip trailing carriage return (Windows line endings)
      if (!line.empty() && line.back() == '\r')
        line.pop_back();

      if (line == "$ParametricNodes") {
        ss << "$Nodes\n";
        in_parametric_nodes = true;
        first_line_after_header = true;
      } else if (line == "$EndParametricNodes") {
        ss << "$EndNodes\n";
        in_parametric_nodes = false;
      } else if (in_parametric_nodes) {
        if (first_line_after_header) {
          // This is the node count line, pass through
          ss << line << "\n";
          first_line_after_header = false;
        } else {
          // Node line: "id x y z [parametric_data...]"
          // Keep only the first 4 fields: id x y z
          std::istringstream iss(line);
          int id;
          double x, y, z;
          iss >> id >> x >> y >> z;
          ss << id << " " << x << " " << y << " " << z << "\n";
        }
      } else {
        ss << line << "\n";
      }
    }

    grid_in.read_msh(ss);

    // Partition the serial mesh for MPI distribution
    GridTools::partition_triangulation(mpi_size, serial_mesh);

    // Build the fully distributed triangulation
    auto construction_data = TriangulationDescription::Utilities::
        create_description_from_triangulation(serial_mesh, MPI_COMM_WORLD);
    mesh.create_triangulation(construction_data);
  }

  double U_mean = 0.0;
  if (dim == 2)
    U_mean = (2.0 / 3.0) * U_m;
  else
    U_mean = (4.0 / 9.0) * U_m; // Check paper equation for specific 3D case

  // Re = (U_mean * D) / nu  ->  nu = (U_mean * D) / Re
  nu = (U_mean * D) / Re;

  pcout << "  Reynolds number: " << Re << std::endl;
  pcout << "  U_max (Inlet param): " << U_m << std::endl;
  pcout << "  U_mean (Reference): " << U_mean << std::endl;
  pcout << "  Cylinder Diameter (D): " << D << std::endl;
  pcout << "  Computed Kinematic viscosity (nu): " << nu << std::endl;
  pcout << "  Time step: " << deltat << std::endl;
  pcout << "  Time scheme: " << to_string(time_scheme) << " (theta=" << theta
        << ")" << std::endl;
  pcout << "  Nonlinear method: " << to_string(nonlinear_method) << std::endl;

  // 1. DoFHandler
  dof_handler.reinit(mesh);
  dof_handler.distribute_dofs(*fe);

  // 2. Blocks enumeration (velocity and pressure)
  std::vector<unsigned int> block_component(dim + 1, 0);
  block_component[dim] = 1; // Pressure is the last block
  DoFRenumbering::component_wise(dof_handler, block_component);

  // 3. DoFs counting
  const std::vector<types::global_dof_index> dofs_per_block =
      DoFTools::count_dofs_per_fe_block(dof_handler, block_component);

  const unsigned int n_u = dofs_per_block[0];
  const unsigned int n_p = dofs_per_block[1];

  pcout << "  Number of active cells: " << mesh.n_active_cells() << std::endl;
  pcout << "  Number of degrees of freedom: " << dof_handler.n_dofs() << " ("
        << n_u << " + " << n_p << ")" << std::endl;

  // 4. IndexSets initialization (parallel management)
  locally_owned_dofs = dof_handler.locally_owned_dofs();
  DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

  // Verification of Boundary IDs (Cause of errors in assembly if mismatch
  // between mesh and code)
  {
    const std::vector<types::boundary_id> ids = mesh.get_boundary_ids();
    pcout << "  Boundary IDs found in mesh: ";
    for (auto id : ids)
      pcout << id << " ";
    pcout << std::endl;

    pcout << "  Expected IDs: Inlet=" << inlet_boundary_id
          << ", Walls=" << wall_boundary_id
          << ", Cylinder=" << cylinder_boundary_id << std::endl;

    // Safety check: if expected IDs are missing, assign them geometrically
    bool has_inlet = false, has_walls = false, has_cylinder = false;
    for (auto id : ids) {
      if (id == inlet_boundary_id)
        has_inlet = true;
      if (id == wall_boundary_id)
        has_walls = true;
      if (id == cylinder_boundary_id)
        has_cylinder = true;
    }

    if (!has_inlet || !has_walls || !has_cylinder) {
      pcout << "  WARNING: Expected boundary IDs not found! "
            << "Assigning boundary IDs geometrically..." << std::endl;

      const double tol = 1e-6;
      const double cx = 0.2;
      const double cy = 0.2;
      const double L = 2.2;
      const double r_cyl = D / 2.0;

      for (const auto &cell : mesh.active_cell_iterators()) {
        if (!cell->is_locally_owned())
          continue;
        for (unsigned int f = 0; f < cell->n_faces(); ++f) {
          if (!cell->face(f)->at_boundary())
            continue;

          const auto center = cell->face(f)->center();

          if constexpr (dim == 2) {
            const double x = center[0];
            const double y = center[1];
            const double dist_cyl =
                std::sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy));

            if (dist_cyl < r_cyl + 0.02)
              cell->face(f)->set_boundary_id(cylinder_boundary_id);
            else if (std::abs(x) < tol)
              cell->face(f)->set_boundary_id(inlet_boundary_id);
            else if (std::abs(x - L) < tol)
              cell->face(f)->set_boundary_id(outlet_boundary_id);
            else
              cell->face(f)->set_boundary_id(wall_boundary_id);
          } else if constexpr (dim == 3) {
            const double x = center[0];
            const double y = center[1];
            const double z = center[2];
            const double dist_cyl =
                std::sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy));

            if (dist_cyl < r_cyl + 0.02)
              cell->face(f)->set_boundary_id(cylinder_boundary_id);
            else if (std::abs(z) < tol)
              cell->face(f)->set_boundary_id(inlet_boundary_id);
            else if (std::abs(z - L) < tol)
              cell->face(f)->set_boundary_id(outlet_boundary_id);
            else
              cell->face(f)->set_boundary_id(wall_boundary_id);
          }
        }
      }

      const std::vector<types::boundary_id> new_ids = mesh.get_boundary_ids();
      pcout << "  Boundary IDs after geometric assignment: ";
      for (auto id : new_ids)
        pcout << id << " ";
      pcout << std::endl;
    }
  }

  block_owned_dofs.resize(2);
  block_owned_dofs[0] = locally_owned_dofs.get_view(0, n_u);
  block_owned_dofs[1] = locally_owned_dofs.get_view(n_u, n_u + n_p);

  block_relevant_dofs.resize(2);
  block_relevant_dofs[0] = locally_relevant_dofs.get_view(0, n_u);
  block_relevant_dofs[1] = locally_relevant_dofs.get_view(n_u, n_u + n_p);

  // 5. Vectors initialization
  // solution_owned: contains only local DoFs (for the solver)
  solution_owned.reinit(block_owned_dofs, MPI_COMM_WORLD);
  system_rhs.reinit(block_owned_dofs, MPI_COMM_WORLD);
  newton_update.reinit(block_owned_dofs, MPI_COMM_WORLD);
  solution_backup.reinit(block_owned_dofs, MPI_COMM_WORLD);

  // solution, solution_old, current_solution: contain also ghost DoFs (for
  // assembly)
  solution.reinit(block_owned_dofs, block_relevant_dofs, MPI_COMM_WORLD);
  solution_old.reinit(block_owned_dofs, block_relevant_dofs, MPI_COMM_WORLD);
  current_solution.reinit(block_owned_dofs, block_relevant_dofs,
                          MPI_COMM_WORLD);

  // For CN: u^{n-1} extrapolation vector
  solution_old_old.reinit(block_owned_dofs, block_relevant_dofs,
                          MPI_COMM_WORLD);

  // Set initial conditions
  solution = 0.0;
  solution_old = 0.0;
  current_solution = 0.0;

  // TODO: Maybe set initial condition here (understand if it's nedeed)

  // 6. Build constraints for Newton updates (homogeneous Dirichlet on velocity)
  //    These are also needed to build a correct sparsity pattern for MPI.
  {
    const FEValuesExtractors::Vector velocities_ext(0);
    const ComponentMask vel_mask = fe->component_mask(velocities_ext);

    newton_constraints.clear();
    newton_constraints.reinit(locally_relevant_dofs);
    VectorTools::interpolate_boundary_values(
        *mapping, dof_handler, inlet_boundary_id,
        Functions::ZeroFunction<dim>(dim + 1), newton_constraints, vel_mask);
    VectorTools::interpolate_boundary_values(
        *mapping, dof_handler, wall_boundary_id,
        Functions::ZeroFunction<dim>(dim + 1), newton_constraints, vel_mask);
    VectorTools::interpolate_boundary_values(
        *mapping, dof_handler, cylinder_boundary_id,
        Functions::ZeroFunction<dim>(dim + 1), newton_constraints, vel_mask);
    newton_constraints.close();
  }

  // 7. Matrices initialization
  //    Block sparsity pattern: pass constraints + distribute for MPI
  //    correctness. Without distribute_sparsity_pattern, off-process matrix
  //    entries are silently dropped by Trilinos during compress(), producing an
  //    incorrect Jacobian when running on multiple MPI ranks.
  {
    const std::vector<types::global_dof_index> dofs_per_block_sizes = {n_u,
                                                                       n_p};
    BlockDynamicSparsityPattern bdsp(dofs_per_block_sizes,
                                     dofs_per_block_sizes);
    DoFTools::make_sparsity_pattern(dof_handler, bdsp, newton_constraints,
                                    false);
    SparsityTools::distribute_sparsity_pattern(
        bdsp, Utilities::MPI::all_gather(MPI_COMM_WORLD, locally_owned_dofs),
        MPI_COMM_WORLD, locally_relevant_dofs);

    system_matrix.reinit(block_owned_dofs, bdsp, MPI_COMM_WORLD);
    pressure_mass.reinit(block_owned_dofs, bdsp, MPI_COMM_WORLD);
    pressure_stiffness.reinit(block_owned_dofs, bdsp, MPI_COMM_WORLD);
  }

  pcout << "Setup complete." << std::endl;
}

template <unsigned int dim> void NavierStokes<dim>::assemble_newton_system() {
  system_matrix = 0;
  system_rhs = 0;
  if (!pressure_mass_assembled)
    pressure_mass = 0;
  if (!pressure_stiffness_assembled)
    pressure_stiffness = 0;

  FEValues<dim> fe_values(*mapping, *fe, *quadrature,
                          update_values | update_gradients | update_hessians |
                              update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q_points = quadrature->size();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_pressure_mass(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_pressure_stiff(dofs_per_cell, dofs_per_cell);
  Vector<double> cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  // Extractors for FEValues (handle vectors and scalars)
  const FEValuesExtractors::Vector velocities(0);
  const FEValuesExtractors::Scalar pressure(dim);

  // Values of the solution at iteration k (current) and at time n-1 (old)
  std::vector<Tensor<1, dim>> current_velocity_values(n_q_points);
  std::vector<Tensor<2, dim>> current_velocity_gradients(n_q_points);
  std::vector<Tensor<1, dim>> current_velocity_laplacians(n_q_points);
  std::vector<double> current_pressure_values(n_q_points);
  std::vector<Tensor<1, dim>> current_pressure_gradients(n_q_points);

  std::vector<Tensor<1, dim>> old_velocity_values(n_q_points);
  std::vector<Tensor<2, dim>> old_velocity_gradients(
      n_q_points); // needed for CN (theta<1)

  // Support vectors for phi_u, grad_phi_u, div_phi_u, phi_p, grad_phi_p
  std::vector<Tensor<1, dim>> phi_u(dofs_per_cell);
  std::vector<Tensor<2, dim>> grad_phi_u(dofs_per_cell);
  std::vector<double> div_phi_u(dofs_per_cell);
  std::vector<double> phi_p(dofs_per_cell);
  std::vector<Tensor<1, dim>> grad_phi_p(
      dofs_per_cell); // for pressure Laplacian

  // Forcing term evaluation vectors (were missing – needed for residual)
  Vector<double> f_val_new(dim + 1);
  Vector<double> f_val_old(dim + 1);

  for (const auto &cell : dof_handler.active_cell_iterators()) {
    if (cell->is_locally_owned()) {
      cell_matrix = 0;
      cell_rhs = 0;
      cell_pressure_mass = 0;
      cell_pressure_stiff = 0;

      fe_values.reinit(cell);

      // Get current solution
      fe_values[velocities].get_function_values(current_solution,
                                                current_velocity_values);
      fe_values[velocities].get_function_gradients(current_solution,
                                                   current_velocity_gradients);
      fe_values[velocities].get_function_laplacians(
          current_solution, current_velocity_laplacians);
      fe_values[pressure].get_function_values(current_solution,
                                              current_pressure_values);
      fe_values[pressure].get_function_gradients(current_solution,
                                                 current_pressure_gradients);

      fe_values[velocities].get_function_values(solution_old,
                                                old_velocity_values);
      fe_values[velocities].get_function_gradients(solution_old,
                                                   old_velocity_gradients);

      for (unsigned int q = 0; q < n_q_points; ++q) {
        const double JxW = fe_values.JxW(q);

        // Precompute basis functions + divergence (avoid recomputing in j-loop)
        for (unsigned int k = 0; k < dofs_per_cell; ++k) {
          phi_u[k] = fe_values[velocities].value(k, q);
          grad_phi_u[k] = fe_values[velocities].gradient(k, q);
          div_phi_u[k] = fe_values[velocities].divergence(k, q);
          phi_p[k] = fe_values[pressure].value(k, q);
          grad_phi_p[k] = fe_values[pressure].gradient(k, q);
        }

        // Evaluate forcing at t^{n+1} and t^n (for theta-method)
        const auto &x_q = fe_values.quadrature_point(q);
        forcing_term->set_time(time);
        forcing_term->vector_value(x_q, f_val_new);
        forcing_term->set_time(time - deltat);
        forcing_term->vector_value(x_q, f_val_old);
        Tensor<1, dim> f_new, f_old;
        for (unsigned int d = 0; d < dim; ++d) {
          f_new[d] = f_val_new[d];
          f_old[d] = f_val_old[d];
        }

        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
          // Residual with theta-method (theta=1 BE, theta=0.5 CN)

          // Time term: (u_k - u_old) / dt * v
          const double time_term =
              (current_velocity_values[q] - old_velocity_values[q]) * phi_u[i] /
              deltat;

          // Implicit part (at current Newton iterate)
          const double conv_impl =
              theta *
              (current_velocity_values[q] * current_velocity_gradients[q]) *
              phi_u[i];
          const double visc_impl =
              theta * nu *
              scalar_product(current_velocity_gradients[q], grad_phi_u[i]);

          // Explicit part (at old time step) — only active for CN (theta<1)
          const double conv_expl =
              (1.0 - theta) *
              (old_velocity_values[q] * old_velocity_gradients[q]) * phi_u[i];
          const double visc_expl =
              (1.0 - theta) * nu *
              scalar_product(old_velocity_gradients[q], grad_phi_u[i]);

          // Pressure term: - p_k * div v (fully implicit)
          const double pres_term = -current_pressure_values[q] * div_phi_u[i];

          // Incompressibility term: - q * div u_k (fully implicit)
          const double div_term =
              -phi_p[i] * trace(current_velocity_gradients[q]);

          cell_rhs(i) += (-time_term - conv_impl - visc_impl - conv_expl -
                          visc_expl - pres_term - div_term) *
                         JxW;

          // Forcing: +theta*f^{n+1} . phi + (1-theta)*f^n . phi
          // (This is part of -R: the forcing has a + sign in the
          //  momentum equation, so it enters the RHS = -R with + sign.)
          cell_rhs(i) += (theta * (f_new * phi_u[i]) +
                          (1.0 - theta) * (f_old * phi_u[i])) *
                         JxW;

          // Newton Jacobian with theta-method
          for (unsigned int j = 0; j < dofs_per_cell; ++j) {
            double val = (phi_u[i] * phi_u[j]) / deltat;

            val += theta * nu * scalar_product(grad_phi_u[i], grad_phi_u[j]);

            // Linearized Convection (implicit fraction)
            val += theta *
                   ((current_velocity_values[q] * grad_phi_u[j]) * phi_u[i] +
                    (phi_u[j] * current_velocity_gradients[q]) * phi_u[i]);

            // Pressure: -(p, div v)
            val -= phi_p[j] * div_phi_u[i];

            // Divergence: -(q, div u)
            val -= phi_p[i] * div_phi_u[j];

            cell_matrix(i, j) += val * JxW;

            // --- SUPG Stabilization (LHS) ---
            // Parametro Tau
            const double h = cell->diameter();
            const double u_mag = current_velocity_values[q].norm();
            // Formula: tau = ((2/dt)^2 + (2|u|/h)^2 + (4nu/h^2)^2)^(-0.5)
            const double tau = 1.0 / std::sqrt(std::pow(2.0 / deltat, 2) +
                                               std::pow(2.0 * u_mag / h, 2) +
                                               std::pow(4.0 * nu / (h * h), 2));

            // Termine SUPG: (u . grad phi_i) * tau * (u . grad phi_j + grad
            // psi_j)
            const double supg_lhs =
                (current_velocity_values[q] * grad_phi_u[i]) * tau *
                (current_velocity_values[q] * grad_phi_u[j] + grad_phi_p[j]);
            cell_matrix(i, j) += supg_lhs * JxW;

            // --- Grad-Div Stabilization (LHS) ---
            // gamma * (div phi_i * div phi_j)
            const double gamma = 0.1;
            const double grad_div_lhs = gamma * (div_phi_u[i] * div_phi_u[j]);
            cell_matrix(i, j) += grad_div_lhs * JxW;

            // Pressure mass matrix for preconditioner
            if (!pressure_mass_assembled)
              cell_pressure_mass(i, j) += phi_p[i] * phi_p[j] * JxW;

            // Pressure stiffness (Laplacian) for Cahouet-Chabard preconditioner
            if (!pressure_stiffness_assembled)
              cell_pressure_stiff(i, j) += grad_phi_p[i] * grad_phi_p[j] * JxW;
          }

          // --- SUPG Stabilization (RHS Residual) ---
          {
            // Recompute tau locally (redundant computation but cleaner scope,
            // valid since q is same)
            const double h = cell->diameter();
            const double u_mag = current_velocity_values[q].norm();
            const double tau = 1.0 / std::sqrt(std::pow(2.0 / deltat, 2) +
                                               std::pow(2.0 * u_mag / h, 2) +
                                               std::pow(4.0 * nu / (h * h), 2));

            // Strong Residual Calculation
            // Time term: (u - u_old)/dt  (Backward Euler approx for residual or
            // consistent with theta?) Linearized residual uses (u - u_old)/dt
            // approx usually. Let's use the same time term as in weak form: (u
            // - u_old)/dt
            Tensor<1, dim> time_res =
                (current_velocity_values[q] - old_velocity_values[q]) / deltat;

            // Convection: u . grad u
            Tensor<1, dim> conv_res =
                current_velocity_values[q] * current_velocity_gradients[q];

            // Pressure: grad p
            Tensor<1, dim> pres_res = current_pressure_gradients[q];

            // Viscosity: -nu Delta u
            Tensor<1, dim> visc_res = -nu * current_velocity_laplacians[q];

            // Forcing: -f (evaluated at t^{n+theta})
            Tensor<1, dim> force_res = -(theta * f_new + (1.0 - theta) * f_old);

            // Total Strong Momentum Residual
            Tensor<1, dim> strong_res =
                time_res + conv_res + pres_res + visc_res + force_res;

            // Subtract (tau * u . grad phi_i) * strong_res from RHS
            // (The RHS vector accumulates -Residual, so we subtract valid SUPG
            // term from valid Residual?
            //  No, RHS = -Residual.
            //  Standard weak form: (R, v) + (R_strong, tau u.grad v) = 0
            //  => (R, v) = - (R_strong, tau u.grad v)
            //  New RHS = - [ (Residual_Weak, v) + (Strong_Res, tau_v) ]
            //  Existing code accumulates into cell_rhs: -Weak_Res_Terms.
            //  So we must also subtract the SUPG term.
            cell_rhs(i) -= (current_velocity_values[q] * grad_phi_u[i]) * tau *
                           strong_res * JxW;
          }
        }
      }

      cell->get_dof_indices(local_dof_indices);

      newton_constraints.distribute_local_to_global(
          cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
      if (!pressure_mass_assembled)
        newton_constraints.distribute_local_to_global(
            cell_pressure_mass, local_dof_indices, pressure_mass);
      if (!pressure_stiffness_assembled)
        newton_constraints.distribute_local_to_global(
            cell_pressure_stiff, local_dof_indices, pressure_stiffness);
    }
  }

  system_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);
  if (!pressure_mass_assembled) {
    pressure_mass.compress(VectorOperation::add);
    pressure_mass_assembled = true;
  }
  if (!pressure_stiffness_assembled) {
    pressure_stiffness.compress(VectorOperation::add);
    pressure_stiffness_assembled = true;
  }
}

template <unsigned int dim> void NavierStokes<dim>::solve_newton_system() {
  const double rhs_norm = system_rhs.l2_norm();

  // Usiamo una tolleranza di 1e-8
  SolverControl solver_control(200, 1e-8 * rhs_norm);

  // Cahouet-Chabard preconditioner:
  //   Velocity block: ILU (robust for convection-dominated)
  //   Pressure block: S^{-1} ≈ -(rho/dt)*K_p^{-1} - theta*nu*M_p^{-1}
  PreconditionBlockTriangular preconditioner;
  preconditioner.initialize(system_matrix.block(0, 0),
                            pressure_mass.block(1, 1),
                            pressure_stiffness.block(1, 1),
                            system_matrix.block(1, 0), nu, rho, deltat, theta);

  // GMRES con restart=150 per sistemi saddle-point
  typename SolverGMRES<TrilinosWrappers::MPI::BlockVector>::AdditionalData
      gmres_data(150);
  SolverGMRES<TrilinosWrappers::MPI::BlockVector> solver(solver_control,
                                                         gmres_data);

  // Solve for newton_update (delta u)
  newton_update = 0.0;
  solver.solve(system_matrix, newton_update, system_rhs, preconditioner);

  pcout << "  GMRES (Newton): " << solver_control.last_step() << " iters"
        << std::endl;

  newton_constraints.distribute(newton_update);
}

template <unsigned int dim>
void NavierStokes<dim>::assemble_linearized_system() {
  system_matrix = 0;
  system_rhs = 0;
  if (!pressure_mass_assembled)
    pressure_mass = 0;
  if (!pressure_stiffness_assembled)
    pressure_stiffness = 0;

  FEValues<dim> fe_values(*mapping, *fe, *quadrature,
                          update_values | update_gradients | update_hessians |
                              update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q_points = quadrature->size();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_pressure_mass(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_pressure_stiff(dofs_per_cell, dofs_per_cell);
  Vector<double> cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  const FEValuesExtractors::Vector velocities(0);
  const FEValuesExtractors::Scalar pressure(dim);

  // Old solution values
  std::vector<Tensor<1, dim>> old_velocity_values(n_q_points);
  std::vector<Tensor<2, dim>> old_velocity_gradients(n_q_points);
  std::vector<Tensor<1, dim>> old_velocity_laplacians(
      n_q_points); // Assuming we can get laplacians of old solution if
                   // needed?
  // Actually we need laplacians of phi_j (trial functions) for LHS
  // consistency, and laplacians of u_old only for RHS time term consistency?
  // For linearized, we need to apply the operator to the trial function.
  // Op(phi_j) = ... - nu * Delta phi_j ...
  // So we need hessians of the FE shape functions.
  std::vector<Tensor<1, dim>> phi_u_lapl(dofs_per_cell);

  std::vector<Tensor<1, dim>> old_old_velocity_values(n_q_points);

  // Basis functions
  std::vector<Tensor<1, dim>> phi_u(dofs_per_cell);
  std::vector<Tensor<2, dim>> grad_phi_u(dofs_per_cell);
  std::vector<double> div_phi_u(dofs_per_cell);
  std::vector<double> phi_p(dofs_per_cell);
  std::vector<Tensor<1, dim>> grad_phi_p(
      dofs_per_cell); // for pressure Laplacian

  // Forcing term evaluation vectors
  Vector<double> f_val_new(dim + 1);
  Vector<double> f_val_old(dim + 1);

  // Build non-homogeneous constraints for this time step
  const ComponentMask velocity_mask = fe->component_mask(velocities);

  system_constraints.clear();
  system_constraints.reinit(locally_relevant_dofs);
  VectorTools::interpolate_boundary_values(*mapping, dof_handler,
                                           inlet_boundary_id, *inlet_velocity,
                                           system_constraints, velocity_mask);
  VectorTools::interpolate_boundary_values(
      *mapping, dof_handler, wall_boundary_id,
      Functions::ZeroFunction<dim>(dim + 1), system_constraints, velocity_mask);
  VectorTools::interpolate_boundary_values(
      *mapping, dof_handler, cylinder_boundary_id,
      Functions::ZeroFunction<dim>(dim + 1), system_constraints, velocity_mask);
  system_constraints.close();

  // Cell loop
  for (const auto &cell : dof_handler.active_cell_iterators()) {
    if (!cell->is_locally_owned())
      continue;

    cell_matrix = 0;
    cell_rhs = 0;
    cell_pressure_mass = 0;
    cell_pressure_stiff = 0;

    fe_values.reinit(cell);

    fe_values[velocities].get_function_values(solution_old,
                                              old_velocity_values);
    fe_values[velocities].get_function_gradients(solution_old,
                                                 old_velocity_gradients);
    fe_values[velocities].get_function_values(solution_old_old,
                                              old_old_velocity_values);

    for (unsigned int q = 0; q < n_q_points; ++q) {
      const double JxW = fe_values.JxW(q);

      // Extrapolated transport velocity
      Tensor<1, dim> u_star;
      if (first_step || second_step || time_scheme == TimeScheme::BackwardEuler)
        u_star = old_velocity_values[q]; // 1st-order: u* = u^n
      else {
        u_star = 2.0 * old_velocity_values[q] -
                 old_old_velocity_values[q]; // 2nd-order: u* = 2u^n - u^{n-1}
        // Safety clamp: se u_star cresce più del 20% rispetto a u^n, ricada
        // su u^n
        const double norm_star = u_star.norm();
        const double norm_old = old_velocity_values[q].norm();
        if (norm_old > 1e-12 && norm_star > 1.2 * norm_old)
          u_star = old_velocity_values[q];
      }

      // Precompute basis functions + divergence
      for (unsigned int k = 0; k < dofs_per_cell; ++k) {
        phi_u[k] = fe_values[velocities].value(k, q);
        grad_phi_u[k] = fe_values[velocities].gradient(k, q);
        div_phi_u[k] = fe_values[velocities].divergence(k, q);
        phi_p[k] = fe_values[pressure].value(k, q);
        grad_phi_p[k] = fe_values[pressure].gradient(k, q);
        // Laplacian of velocity shape function
        // We need to fetch it from fe_values using the shape_hessian or some
        // component method? FEValues::shape_hessian gives full hessian. Trace
        // is Laplacian. Helper:

        // Vector FE: hessian(k,q) returns Tensor<3,dim> if k is vector
        // component index? Actually fe_values[velocities].hessian(k,q)
        // returns Tensor<2,dim> which is the hessian of the scalar shape
        // function? No, velocities is a vector extractor. Let's use the
        // explicit computed Laplacian if possible, but FESystem shape
        // functions are tricky. For now, let's assume -nu*Delta u is small
        // and neglect it in the LHS operator for SUPG consistency OR compute
        // it properly. Given the user instructions, they didn't ask for full
        // consistency in LHS, just "SUPG". "1. SUPG ... Formula: ... (u_conv
        // * grad(phi_i)) * tau * (u_conv * grad(phi_i))" That formula
        // corresponds to Convection-Convection only. I will stick to what the
        // user asked for in LHS to avoid compilation headaches with shape
        // laplacians.
      }

      // Evaluate forcing term at t^{n+1} and t^n
      const auto &x_q = fe_values.quadrature_point(q);
      forcing_term->set_time(time);
      forcing_term->vector_value(x_q, f_val_new);
      forcing_term->set_time(time - deltat);
      forcing_term->vector_value(x_q, f_val_old);
      Tensor<1, dim> f_new, f_old;
      for (unsigned int d = 0; d < dim; ++d) {
        f_new[d] = f_val_new[d];
        f_old[d] = f_val_old[d];
      }

      for (unsigned int i = 0; i < dofs_per_cell; ++i) {
        // RHS: explicit contributions from time n
        // (1/dt)(u^n . phi)
        const double rhs_mass =
            (1.0 / deltat) * (old_velocity_values[q] * phi_u[i]);

        // -(1-theta) * nu * (grad u^n : grad phi)
        const double rhs_visc =
            -(1.0 - theta) * nu *
            scalar_product(old_velocity_gradients[q], grad_phi_u[i]);

        // -(1-theta) * ((u^n . grad)u^n . phi)
        const double rhs_conv =
            -(1.0 - theta) *
            ((old_velocity_values[q] * old_velocity_gradients[q]) * phi_u[i]);

        cell_rhs(i) += (rhs_mass + rhs_visc + rhs_conv) * JxW;

        // Forcing term: +theta * f^{n+1} . phi + (1-theta) * f^n . phi
        cell_rhs(i) +=
            (theta * (f_new * phi_u[i]) + (1.0 - theta) * (f_old * phi_u[i])) *
            JxW;

        // --- SUPG Stabilization (Linearized) ---
        // Calculate Tau
        const double h = cell->diameter();
        const double u_mag = u_star.norm();
        const double tau = 1.0 / std::sqrt(std::pow(2.0 / deltat, 2) +
                                           std::pow(2.0 * u_mag / h, 2) +
                                           std::pow(4.0 * nu / (h * h), 2));

        // SUPG Test Function: w_SUPG = tau * (u_star . grad phi_i)
        // SUPG Test Function vector: w_SUPG = tau * (u_star . grad phi_i)
        Tensor<1, dim> supg_test_vec = tau * (u_star * grad_phi_u[i]);

        // --- SUPG RHS Consistency ---
        // Forcing term (theta-method)
        Tensor<1, dim> forcing_rhs = (theta * f_new + (1.0 - theta) * f_old);

        // RHS contribution: (supg_test_vec) dot (RHS_source)
        // RHS_source = f + u_old/dt
        Tensor<1, dim> rhs_source =
            forcing_rhs + old_velocity_values[q] / deltat;

        cell_rhs(i) += (supg_test_vec * rhs_source) * JxW;

        // LHS: matrix entries
        for (unsigned int j = 0; j < dofs_per_cell; ++j) {
          // Mass: (1/dt)(phi_j . phi_i)
          double val = (1.0 / deltat) * (phi_u[i] * phi_u[j]);

          // Viscosity (implicit fraction): theta * nu * (grad phi_j : grad
          // phi_i)
          val += theta * nu * scalar_product(grad_phi_u[i], grad_phi_u[j]);

          // Semi-implicit convection: theta * (u* . grad phi_j) . phi_i
          val += theta * (u_star * grad_phi_u[j]) * phi_u[i];

          // Pressure: -(psi_j, div phi_i)
          val -= phi_p[j] * div_phi_u[i];

          // Divergence: -(psi_i, div phi_j)
          val -= phi_p[i] * div_phi_u[j];

          cell_matrix(i, j) += val * JxW;

          // --- SUPG Stabilization (Linearized LHS) ---
          // Added term: (supg_test_vec, LinearOperator(phi_j))
          // LinearOperator(phi_j) approx: phi_j/dt + u*.grad phi_j + grad
          // psi_j Note: neglecting -nu Delta phi_j (consistent with user
          // request)

          // Re-calculate supg_test_vec (needed here as i is outer loop, but j
          // changes) Wait, i is fixed in this inner loop.
          Tensor<1, dim> supg_test_vec = tau * (u_star * grad_phi_u[i]);

          Tensor<1, dim> op_phi_j = phi_u[j] / deltat;
          op_phi_j += u_star * grad_phi_u[j]; // Convection
          // op_phi_j += grad_phi_p[j]; // Pressure gradient? phi_p is scalar
          // pressure shape. Wait. j iterates dofs_per_cell. If j corresponds
          // to velocity dof, phi_u[j] is nonzero, phi_p[j] is zero? Yes. But
          // grad_phi_p[j] is only non-zero if j is pressure dof. But op_phi_j
          // vector is momentum equation residual. If j is pressure dof, then
          // phi_u[j]=0. The term is grad psi_j. So we strictly add:

          // 1. Time + Convection part (acting on velocity dofs)
          cell_matrix(i, j) +=
              (supg_test_vec * (phi_u[j] / deltat + u_star * grad_phi_u[j])) *
              JxW;

          // 2. Pressure Gradient part (acting on pressure dofs)
          // If j is pressure dof, grad_phi_p[j] is the gradient.
          // (supg_test_vec * grad_phi_p[j])
          // Note: grad_phi_p[j] is vector.
          cell_matrix(i, j) += (supg_test_vec * grad_phi_p[j]) * JxW;

          // --- Grad-Div Stabilization ---
          // gamma * (div phi_i * div phi_j)
          const double gamma = 0.1;
          cell_matrix(i, j) += gamma * (div_phi_u[i] * div_phi_u[j]) * JxW;

          // Pressure mass for preconditioner
          if (!pressure_mass_assembled)
            cell_pressure_mass(i, j) += phi_p[i] * phi_p[j] * JxW;

          // Pressure stiffness (Laplacian) for Cahouet-Chabard
          if (!pressure_stiffness_assembled)
            cell_pressure_stiff(i, j) += grad_phi_p[i] * grad_phi_p[j] * JxW;
        }
      }
    }

    cell->get_dof_indices(local_dof_indices);

    system_constraints.distribute_local_to_global(
        cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
    if (!pressure_mass_assembled)
      system_constraints.distribute_local_to_global(
          cell_pressure_mass, local_dof_indices, pressure_mass);
    if (!pressure_stiffness_assembled)
      system_constraints.distribute_local_to_global(
          cell_pressure_stiff, local_dof_indices, pressure_stiffness);
  }

  system_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);
  if (!pressure_mass_assembled) {
    pressure_mass.compress(VectorOperation::add);
    pressure_mass_assembled = true;
  }
  if (!pressure_stiffness_assembled) {
    pressure_stiffness.compress(VectorOperation::add);
    pressure_stiffness_assembled = true;
  }
}

template <unsigned int dim> bool NavierStokes<dim>::solve_linear_system() {
  const double rhs_norm = system_rhs.l2_norm();
  // Use 1e-8 relative tolerance
  SolverControl solver_control(200, std::max(1e-8 * rhs_norm, 1e-12));

  PreconditionBlockTriangular preconditioner;
  preconditioner.initialize(system_matrix.block(0, 0),
                            pressure_mass.block(1, 1),
                            pressure_stiffness.block(1, 1),
                            system_matrix.block(1, 0), nu, rho, deltat, theta);

  typename SolverGMRES<TrilinosWrappers::MPI::BlockVector>::AdditionalData
      gmres_data(150);
  SolverGMRES<TrilinosWrappers::MPI::BlockVector> solver(solver_control,
                                                         gmres_data);

  solution_owned = 0.0;

  bool converged = true;
  try {
    solver.solve(system_matrix, solution_owned, system_rhs, preconditioner);
  } catch (const SolverControl::NoConvergence &e) {
    converged = false;
    pcout << "  WARNING: GMRES did NOT converge after "
          << solver_control.last_step() << " iterations, "
          << "residual = " << solver_control.last_value() << std::endl;
    // Continue with the best approximation we have
  }

  system_constraints.distribute(solution_owned);

  if (converged)
    pcout << "  GMRES: " << solver_control.last_step() << " iters" << std::endl;

  return converged;
}

// Compute pressure difference between front and back of the cylinder.
// Robust MPI handling: tracks which ranks found the point to avoid
// silent errors (no rank finds it) or double-counting (multiple ranks find
// it).
template <unsigned int dim>
double NavierStokes<dim>::compute_pressure_difference() {
  Point<dim> p_front, p_end;
  if (dim == 2) {
    p_front = Point<dim>(0.15, 0.2);
    p_end = Point<dim>(0.25, 0.2);
  } else {
    p_front = Point<dim>(0.205, 0.2, 0.40);
    p_end = Point<dim>(0.205, 0.2, 0.50);
  }

  // Helper lambda: evaluate pressure at a point robustly across MPI ranks
  auto evaluate_pressure = [&](const Point<dim> &pt) -> double {
    double local_press = 0.0;
    int found = 0;
    try {
      Vector<double> value(dim + 1);
      VectorTools::point_value(*mapping, dof_handler, current_solution, pt,
                               value);
      local_press = value[dim];
      found = 1;
    } catch (...) {
      local_press = 0.0;
      found = 0;
    }

    // Gather across all ranks
    const int total_found = Utilities::MPI::sum(found, MPI_COMM_WORLD);
    const double global_press =
        Utilities::MPI::sum(local_press, MPI_COMM_WORLD);

    if (total_found > 0)
      return global_press / total_found;
    else {
      pcout << "  WARNING: pressure evaluation point not found by any rank!"
            << std::endl;
      return 0.0;
    }
  };

  return evaluate_pressure(p_front) - evaluate_pressure(p_end);
}

// TODO we have to check if the direction of the flux in 3D is really in the z
// direction
// TODO in this case is set different from the paper (where it is along the x
// direction)
template <unsigned int dim>
void NavierStokes<dim>::compute_lift_drag(double &drag_coeff,
                                          double &lift_coeff) const {
  // Compute forces on the cylinder
  // F = Integral_S (sigma * n) dS
  // sigma = -pI + nu * (grad u + grad u^T)

  double force_x = 0.0;
  double force_y = 0.0;
  double force_z = 0.0; // only for 3D

  const QGaussSimplex<dim - 1> face_quadrature_formula(degree_velocity + 1);

  FEFaceValues<dim> fe_face_values(
      *mapping, *fe, face_quadrature_formula,
      update_values | update_gradients | update_quadrature_points |
          update_JxW_values | update_normal_vectors);

  const FEValuesExtractors::Vector velocities(0);
  const FEValuesExtractors::Scalar pressure(dim);

  std::vector<Tensor<1, dim>> velocity_values(face_quadrature_formula.size());
  std::vector<Tensor<2, dim>> velocity_gradients(
      face_quadrature_formula.size());
  std::vector<double> pressure_values(face_quadrature_formula.size());

  for (const auto &cell : dof_handler.active_cell_iterators()) {
    if (cell->is_locally_owned()) {
      for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f) {
        if (cell->face(f)->at_boundary() &&
            cell->face(f)->boundary_id() == cylinder_boundary_id) {
          fe_face_values.reinit(cell, f);

          fe_face_values[velocities].get_function_values(current_solution,
                                                         velocity_values);
          fe_face_values[velocities].get_function_gradients(current_solution,
                                                            velocity_gradients);
          fe_face_values[pressure].get_function_values(current_solution,
                                                       pressure_values);

          for (unsigned int q = 0; q < face_quadrature_formula.size(); ++q) {
            const double p = pressure_values[q];
            const Tensor<2, dim> grad_u = velocity_gradients[q];
            const Tensor<1, dim> normal = fe_face_values.normal_vector(q);
            const double JxW = fe_face_values.JxW(q);

            // Stress tensor: sigma = -pI + rho*nu*(grad_u + grad_u^T)
            // Note: rho=1 in the code, but we use the member variable rho for
            // correctness
            Tensor<2, dim> stress;
            for (unsigned int i = 0; i < dim; ++i)
              stress[i][i] = -p;

            // Viscous part (symmetric tensor)
            // stress += rho * nu * (grad_u + transpose(grad_u));
            // In deal.II:
            stress += rho * nu * (grad_u + transpose(grad_u));

            // Local force = stress * normal
            Tensor<1, dim> force_loc = stress * normal;

            force_x += force_loc[0] * JxW;
            force_y += force_loc[1] * JxW; // Lift direction
            if constexpr (dim == 3)
              force_z += force_loc[2] * JxW;
          }
        }
      }
    }
  }

  force_x = Utilities::MPI::sum(force_x, MPI_COMM_WORLD);
  force_y = Utilities::MPI::sum(force_y, MPI_COMM_WORLD);
  if constexpr (dim == 3)
    force_z = Utilities::MPI::sum(force_z, MPI_COMM_WORLD);

  // Cd = 2 * Fd / (rho * U_mean^2 * D * L)
  // where L = 1 (2D) or H (3D)
  // The factor 2 is required by the standard Schaefer-Turek benchmark.

  double U_mean = 0.0;
  if (dim == 2)
    U_mean = (2.0 / 3.0) * U_m;
  else
    U_mean = (4.0 / 9.0) * U_m;

  // Reference area. 2D: D (cross-sectional length). 3D: D*H
  double ref_area = D;
  if (dim == 3)
    ref_area = D * H;

  // Schaefer-Turek: C_D = 2*F_D/(rho*U_mean^2*ref_area)
  const double den = 0.5 * rho * U_mean * U_mean * ref_area;

  if constexpr (dim == 2) {
    drag_coeff = force_x / den;
    lift_coeff = force_y / den;
  } else if constexpr (dim == 3) {
    drag_coeff = force_z / den;
    lift_coeff = force_y / den;
  }
}

template <unsigned int dim>
void NavierStokes<dim>::output(const unsigned int time_step) {
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);

  // Partition vectors for output
  std::vector<std::string> solution_names(dim, "velocity");
  solution_names.push_back("pressure");

  std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
          dim, DataComponentInterpretation::component_is_part_of_vector);
  data_component_interpretation.push_back(
      DataComponentInterpretation::component_is_scalar);

  data_out.add_data_vector(current_solution, solution_names,
                           DataOut<dim>::type_dof_data,
                           data_component_interpretation);

  // Add subdomain ids for MPI debugging
  Vector<float> subdomain(mesh.n_active_cells());
  for (unsigned int i = 0; i < subdomain.size(); ++i)
    subdomain(i) = mesh.locally_owned_subdomain();
  data_out.add_data_vector(subdomain, "subdomain");

  data_out.build_patches(*mapping);

  data_out.write_vtu_with_pvtu_record("./", "solution", time_step,
                                      MPI_COMM_WORLD, 4);

  // Add this time step to PVD records and write the collection file
  // The PVTU filename format is solution_XXXXX.pvtu (5-digit zero-padded)
  std::ostringstream pvtu_filename;
  pvtu_filename << "solution_" << std::setfill('0') << std::setw(5) << time_step
                << ".pvtu";

  pvd_records.push_back({time, pvtu_filename.str()});
  write_pvd_file();
}

template <unsigned int dim> void NavierStokes<dim>::write_pvd_file() const {
  // Only rank 0 writes the PVD file
  if (mpi_rank != 0)
    return;

  std::ofstream pvd_file("solution.pvd");
  pvd_file << "<?xml version=\"1.0\"?>\n";
  pvd_file << "<VTKFile type=\"Collection\" version=\"0.1\">\n";
  pvd_file << "  <Collection>\n";

  for (const auto &record : pvd_records) {
    pvd_file << "    <DataSet timestep=\"" << record.first << "\" file=\""
             << record.second << "\"/>\n";
  }

  pvd_file << "  </Collection>\n";
  pvd_file << "</VTKFile>\n";
  pvd_file.close();
}

template <unsigned int dim> void NavierStokes<dim>::run() {
  setup();

  // Solution initialization from initial condition
  inlet_velocity->set_time(0.0);
  initial_condition->set_time(0.0);

  VectorTools::interpolate(*mapping, dof_handler, *initial_condition,
                           solution_owned);
  solution = solution_owned;
  solution_old = solution;
  solution_old_old = solution;
  current_solution = solution;
  first_step = true;
  second_step = true;

  time = 0.0;
  unsigned int time_step = 0;

  // File for benchmark quantities
  std::ofstream forces_file;
  if (mpi_rank == 0) {
    forces_file.open("forces.txt");
    forces_file << "Time\tCd\tCl\tDeltaP" << std::endl;
  }

  // Output initial condition (time_step = 0)
  output(time_step);

  while (time < T) {
    time += deltat;
    time_step++;

    // For the very first step with Crank-Nicolson, override to
    // Backward Euler (theta=1).  CN at the first step amplifies the
    // discontinuity between the zero IC and the inlet BC, creating
    // 2nd-order oscillations in velocity and pressure that often
    // prevent GMRES convergence.  BE damps them out.  After this
    // single step we restore the user's theta.
    const double theta_save = theta;
    if (first_step && time_scheme == TimeScheme::CrankNicolson) {
      theta = 1.0;
      pcout << " (using BE for first step)";
    }

    pcout << "Time step " << time_step << " at t=" << time << std::flush;

    // Update BC and forcing to t^{n+1}
    inlet_velocity->set_time(time);
    forcing_term->set_time(time);

    auto wall_start = std::chrono::high_resolution_clock::now();

    // Newton Approach
    if (nonlinear_method == NonlinearMethod::Newton) {
      // Lift non-homogeneous BCs onto current_solution
      {
        const FEValuesExtractors::Vector velocities(0);
        const ComponentMask velocity_mask = fe->component_mask(velocities);

        std::map<types::global_dof_index, double> boundary_values;

        VectorTools::interpolate_boundary_values(
            *mapping, dof_handler, inlet_boundary_id, *inlet_velocity,
            boundary_values, velocity_mask);
        VectorTools::interpolate_boundary_values(
            *mapping, dof_handler, wall_boundary_id,
            Functions::ZeroFunction<dim>(dim + 1), boundary_values,
            velocity_mask);
        VectorTools::interpolate_boundary_values(
            *mapping, dof_handler, cylinder_boundary_id,
            Functions::ZeroFunction<dim>(dim + 1), boundary_values,
            velocity_mask);

        solution_owned = current_solution;
        for (const auto &[dof, val] : boundary_values) {
          if (locally_owned_dofs.is_element(dof))
            solution_owned(dof) = val;
        }
        current_solution = solution_owned;
      }

      // Newton iteration loop
      double residual_norm = 1e10;
      double previous_residual = 1e10;
      unsigned int newton_iter = 0;
      double damping = 1.0;

      while (residual_norm > newton_tolerance &&
             newton_iter < newton_max_iterations) {
        assemble_newton_system();

        residual_norm = system_rhs.l2_norm();
        pcout << " [" << newton_iter << ": " << residual_norm;
        if (damping < 1.0 - 1e-12)
          pcout << " a=" << damping;
        pcout << "]" << std::flush;

        if (residual_norm < newton_tolerance)
          break;

        // Adaptive damping (Armijo-like)
        if (newton_iter > 0 && residual_norm > 0.99 * previous_residual)
          damping = std::max(0.05, damping * 0.5);
        else if (residual_norm < 0.5 * previous_residual &&
                 damping < 1.0 - 1e-12)
          damping = std::min(1.0, damping * 1.5);
        previous_residual = residual_norm;

        solution_backup = solution_owned;

        bool linear_solve_ok = true;
        try {
          solve_newton_system();
        } catch (const std::exception &) {
          pcout << "(linfail)" << std::flush;
          linear_solve_ok = false;
          damping = std::max(0.05, damping * 0.25);
        }

        // Apply Newton update with damping
        solution_owned = current_solution;
        solution_owned.add(damping, newton_update);
        current_solution = solution_owned;

        // Backtracking on linear failure
        if (!linear_solve_ok) {
          assemble_newton_system();
          const double new_res = system_rhs.l2_norm();
          if (new_res > 2.0 * residual_norm) {
            solution_owned = solution_backup;
            current_solution = solution_owned;
            damping = std::max(0.01, damping * 0.5);
            solution_owned.add(damping, newton_update);
            current_solution = solution_owned;
          }
        }

        newton_iter++;
      }

      pcout << " Newton: " << newton_iter << " iters, res=" << residual_norm;
      if (residual_norm > newton_tolerance)
        pcout << " WARNING: Newton did NOT converge!";
      pcout << std::endl;
    }
    // Linearized Approach
    else {
      // Adaptive time-stepping con checkpointing
      constexpr int max_substeps = 4; // max dimezzamenti dt consentiti

      // Salva lo stato prima di tentare il passo
      TrilinosWrappers::MPI::BlockVector solution_checkpoint = solution_old;
      TrilinosWrappers::MPI::BlockVector solution_old_old_checkpoint =
          solution_old_old;
      const bool first_step_checkpoint = first_step;

      double dt_attempt = deltat;
      bool step_ok = false;
      int substep = 0;

      while (!step_ok && substep <= max_substeps) {
        if (substep > 0) {
          // Dimezza dt e ripristina lo stato al checkpoint
          dt_attempt *= 0.5;
          solution_old = solution_checkpoint;
          solution_old_old = solution_old_old_checkpoint;
          first_step = first_step_checkpoint;
          pcout << "  Retrying with dt=" << dt_attempt << " (attempt "
                << substep + 1 << ")" << std::endl;
        }

        // Sostituisce temporaneamente deltat per l'assembly
        const double deltat_save = deltat;
        deltat = dt_attempt;

        // Tenta con CN + u_star di ordine 2
        assemble_linearized_system();
        bool gmres_ok = solve_linear_system();

        if (!gmres_ok && substep == 0) {
          // Primo fallback: stessa dt ma BE + u_star primo ordine
          pcout << "  Fallback to BE + 1st-order..." << std::endl;
          const double theta_save_inner = theta;
          const bool fs_save = first_step;
          theta = 1.0;
          first_step = true; // forza u_star = u^n
          assemble_linearized_system();
          gmres_ok = solve_linear_system();
          theta = theta_save_inner;
          first_step = fs_save;
        }

        deltat = deltat_save;

        if (gmres_ok) {
          step_ok = true;
          if (substep > 0) {
            pcout << "  Step accepted with reduced dt=" << dt_attempt
                  << std::endl;
          }
        } else {
          substep++;
        }
      }

      if (!step_ok) {
        // Ultimo tentativo disperato: ripristina checkpoint e usa BE puro
        pcout << "  CRITICAL: all attempts failed. "
              << "Restoring checkpoint and forcing BE dt=" << dt_attempt
              << std::endl;
        solution_old = solution_checkpoint;
        solution_old_old = solution_old_old_checkpoint;
        first_step = first_step_checkpoint;
        const double theta_save_inner = theta;
        const bool fs_save = first_step;
        const double deltat_save = deltat;
        theta = 1.0;
        first_step = true;
        deltat = dt_attempt;
        assemble_linearized_system();
        solve_linear_system(); // accettiamo comunque il risultato
        theta = theta_save_inner;
        first_step = fs_save;
        deltat = deltat_save;
      }

      current_solution = solution_owned;
    }

    {
      auto wall_end = std::chrono::high_resolution_clock::now();
      const double wall_sec =
          std::chrono::duration<double>(wall_end - wall_start).count();
      pcout << "  Wall time: " << wall_sec << " s" << std::endl;
    }

    // Shift time levels
    solution_old_old = solution_old;
    solution_old = current_solution;
    second_step = first_step; // second_step = true solo durante il 2° step
    first_step = false;

    // Restore original theta after first step override
    theta = theta_save;

    // Benchmark quantities
    double drag = 0.0, lift = 0.0;
    compute_lift_drag(drag, lift);
    const double delta_p = compute_pressure_difference();

    pcout << "  Cd=" << drag << "  Cl=" << lift << "  dP=" << delta_p
          << std::endl;

    if (mpi_rank == 0) {
      forces_file << time << "\t" << drag << "\t" << lift << "\t" << delta_p
                  << std::endl;
      forces_file.flush();
    }

    // VTU output at every time step (for crash safety)
    output(time_step);
  }

  pcout << "===============================================" << std::endl;
  pcout << "Simulation complete." << std::endl;
}

// Explicit instantiation
template class NavierStokes<2>;
template class NavierStokes<3>;